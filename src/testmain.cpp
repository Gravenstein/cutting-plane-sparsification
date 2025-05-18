#include <scip/cutsel_hybrid.h>
#include <scip/pub_lp.h>
#include <scip/pub_message.h>
#include <scip/pub_sepa.h>
#include <scip/scip_cut.h>
#include <scip/scip_lp.h>
#include <scip/scip_message.h>
#include <scip/scip_numerics.h>
#include <scip/scip_param.h>
#include <scip/type_cutsel.h>
#include <scip/type_lp.h>
#include <scip/type_misc.h>
#include <scip/type_reopt.h>
#include <scip/type_result.h>
#include <scip/type_retcode.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/internal/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "objscip/objscip.h"
#include "objscip/objscipdefplugins.h"
// TODO Add to SCIP MR.
#include "objscip/objtable.h"
#include "scip/struct_scip.h"
#include "sparsifier.h"
#include "statistics.h"
#include "status_macros.h"
#include "sys/stat.h"
#include "sys/types.h"

ABSL_FLAG(bool, default_config, false,
          "Use default SCIP config without any custom additions.");

ABSL_FLAG(std::string, problem_name, "",
          "Name of the problem to solve. Determines default file names.");
ABSL_FLAG(std::string, config_file, "",
          "File with SCIP settings to use (.set).");
ABSL_FLAG(std::string, problem_file, "",
          "File with the problem to solve. See SCIP documentation for "
          "supported formats. If empty, the filename is guessed from the "
          "problem name.");
ABSL_FLAG(std::string, solution_file, "",
          "File with the optimal solution for the problem. See SCIP "
          "documentation for supported formats. If empty, the filename is "
          "guessed from the problem name.");

ABSL_FLAG(std::string, result_dir, "",
          "All result files are placed in this directory.");
ABSL_FLAG(int, seed, -1, "SCIP seed.");
ABSL_FLAG(int, max_selected_nnz, -1,
          "Maximum number of nonzeros in selected cuts.");

std::string ExtractRowOriginName(SCIP* scip, SCIP_ROW* row) {
  SCIP_SEPA* sepa = SCIProwGetOriginSepa(row);
  if (sepa != nullptr) return SCIPsepaGetName(sepa);
  SCIP_CONS* cons = SCIProwGetOriginCons(row);
  if (cons != nullptr) return SCIPconsGetName(cons);
  SCIP_CONSHDLR* conshdlr = SCIProwGetOriginConshdlr(row);
  if (conshdlr != nullptr) return SCIPconshdlrGetName(conshdlr);
  return absl::StrFormat("unknown [%d]", SCIProwGetOrigintype(row));
}

bool IsDirectory(const std::string& name) {
  struct stat info;
  return stat(name.c_str(), &info) == 0 && info.st_mode & S_IFDIR;
}

struct CutStatistics {
  int index = -1;
  std::string name;
  std::string origin;
  std::string sparsification;
  bool local;

  int nnz;
  double density;
  double sum_norm;

  double efficacy;
  double directed_cutoff_distance;

  CutStatistics(SCIP* scip, SCIP_ROW* cut) {
    // We assume that we have injected the optimal solution.
    SCIP_Sol* opt = SCIPgetBestSol(scip);
    DCHECK_NE(opt, nullptr);

    index = SCIProwGetIndex(cut);
    name = SCIProwGetName(cut);
    origin = ExtractRowOriginName(scip, cut);
    local = SCIProwIsLocal(cut);
    nnz = SCIProwGetNNonz(cut);
    density = nnz / static_cast<double>(SCIPgetNVars(scip));
    sum_norm = SCIProwGetSumNorm(cut);
    efficacy = SCIPgetCutEfficacy(scip, nullptr, cut);
    directed_cutoff_distance = SCIPgetCutLPSolCutoffDistance(scip, opt, cut);
  }

  static std::string CSVHeader() {
    return "index,name,origin,sparsification,local,nnz,density "
           "[%],sum_norm,efficacy,directed_cutoff_distance,";
  }

  std::string ToCSVFormat() const {
    return absl::StrFormat("%d,%s,%s,%s,%d,%d,%0.2lf,%0.2lf,%0.4lf,%0.4lf,",
                           index, name, origin, sparsification, local, nnz,
                           density * 100, sum_norm, efficacy,
                           directed_cutoff_distance);
  }
};

class CutByOriginTable : public scip::ObjTable {
 public:
  explicit CutByOriginTable(SCIP* scip, std::string original_dump_file,
                            std::string sparsified_dump_file)
      : scip::ObjTable(scip, "CutByOriginTable",
                       "Summarizes cut statistics by origin.", 20000,
                       SCIP_STAGE_SOLVING),
        original_dump_file_(std::move(original_dump_file)),
        sparsified_dump_file_(std::move(sparsified_dump_file)) {}

  SCIP_DECL_TABLEOUTPUT(scip_output) final {
    SCIPmessageFPrintInfo(
        scip->messagehdlr, file,
        "%-19s: %12s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
        "Sparsification", "total", "local", "eff", "nnz", "density", "Stotal",
        "Slocal", "Seff", "Snnz", "Sdensity");
    for (const std::string& origin : all_origins_) {
      SCIPmessageFPrintInfo(scip->messagehdlr, file,
                            "  %-17s:", origin.c_str());
      PrintStats(scip, file, original_stats_, origin);
      PrintStats(scip, file, sparsified_stats_, origin);
      SCIPmessageFPrintInfo(scip->messagehdlr, file, "\n");
    }
    SCIPmessageFPrintInfo(scip->messagehdlr, file, "  %-17s:", "Aggregated");
    PrintStats(scip, file, original_stats_);
    PrintStats(scip, file, sparsified_stats_);
    SCIPmessageFPrintInfo(scip->messagehdlr, file, "\n");
    return SCIP_OKAY;
  }

  SCIP_DECL_TABLEEXIT(scip_exit) final {
    DumpStatisticsToFile(scip, original_stats_, original_dump_file_);
    DumpStatisticsToFile(scip, sparsified_stats_, sparsified_dump_file_);
    return SCIP_OKAY;
  }

  void AddStats(const CutStatistics& original,
                const CutStatistics& sparsified) {
    original_stats_.push_back(original);
    sparsified_stats_.push_back(sparsified);
    all_origins_.insert(original.origin);
    all_origins_.insert(sparsified.origin);
  }

 private:
  std::vector<CutStatistics> original_stats_;
  std::vector<CutStatistics> sparsified_stats_;
  absl::flat_hash_set<std::string> all_origins_;

  std::string original_dump_file_;
  std::string sparsified_dump_file_;

  static void PrintStats(SCIP* scip, FILE* file,
                         std::vector<CutStatistics>& stats,
                         const std::string& origin = "") {
    const auto end =
        origin.empty() ? stats.end()
                       : std::partition(stats.begin(), stats.end(),
                                        [&origin](const CutStatistics& stats) {
                                          return stats.origin == origin;
                                        });
    const int n = std::distance(stats.begin(), end);
    const int n_local =
        std::count_if(stats.begin(), end,
                      [](const CutStatistics& stats) { return stats.local; });
    const double efficacy =
        std::accumulate(stats.begin(), end, 0.0,
                        [](double v, const CutStatistics& stat) {
                          return v + stat.efficacy;
                        }) /
        n;
    const double nnz = std::accumulate(stats.begin(), end, 0.0,
                                       [](double v, const CutStatistics& stat) {
                                         return v + stat.nnz;
                                       }) /
                       n;
    const double density =
        std::accumulate(stats.begin(), end, 0.0,
                        [](double v, const CutStatistics& stat) {
                          return v + stat.density;
                        }) /
        n;
    SCIPmessageFPrintInfo(scip->messagehdlr, file,
                          " %12d %12d %12.4lf %12.2lf %12.2lf", n, n_local,
                          efficacy, nnz, density * 100);
  }
};

class NonzeroTable : public scip::ObjTable {
 public:
  explicit NonzeroTable(SCIP* scip)
      : scip::ObjTable(scip, "NonzeroTable",
                       "Counts the number of nonzeros in rows and cuts.", 19000,
                       SCIP_STAGE_SOLVING) {}

  SCIP_DECL_TABLEOUTPUT(scip_output) final {
    SCIPmessageFPrintInfo(scip->messagehdlr, file,
                          "Nonzeros in selected cuts: %d\n",
                          selected_cut_nonzeros_);
    return SCIP_OKAY;
  }

  SCIP_DECL_TABLEINIT(scip_init) final {
    selected_cut_nonzeros_ = 0;
    return SCIP_OKAY;
  }

  SCIP_DECL_TABLEEXIT(scip_exit) final { return SCIP_OKAY; }

  void AddCut(SCIP_ROW* row) { selected_cut_nonzeros_ += SCIProwGetNNonz(row); }

 private:
  int selected_cut_nonzeros_ = 0;
};

class CutStatisticsCollector : public scip::ObjCutsel {
 private:
  // Pauses the SCIP solve time and tracks the time using the selector clock
  // instead. Uses RAII to stop/start the clocks.
  // Uses auto return type, so must be defined before it is used.
  [[nodiscard]] auto MeassureTime() {
    SCIPstopSolvingTime(scip_);
    return absl::MakeCleanup([scip = scip_]() { SCIPstartSolvingTime(scip); });
  }

 public:
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  static constexpr char kName[] = "cut_statistics_collector";
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  static constexpr char kDescription[] =
      "Cut selector that collects statistics on the generated cuts. Actual "
      "selection is deferred to the default strategy.";
  static constexpr int kPriority = 1000000;

  explicit CutStatisticsCollector(
      SCIP* scip, std::unique_ptr<SparsifierInterface> sparsifier,
      CutByOriginTable* table, NonzeroTable* nnz_table)
      : scip::ObjCutsel(scip, kName, kDescription, kPriority),
        sparsifier_(std::move(sparsifier)),
        table_(table),
        nnz_table_(nnz_table) {
    CHECK_NE(sparsifier_.get(), nullptr);
    CHECK_NE(table_, nullptr);
    CHECK_EQ(
        SCIPaddBoolParam(scip, "sparsifier/replace_original",
                         "Should the sparsified cut replace the original cut?",
                         &replace_original_, true, true, nullptr, nullptr),
        SCIP_OKAY);
    CHECK_EQ(SCIPaddBoolParam(scip, "sparsifier/call_sparsifier",
                              "Should the sparsifier be called?",
                              &call_sparsifier_, false, true, nullptr, nullptr),
             SCIP_OKAY);
    CHECK_EQ(
        SCIPaddBoolParam(scip, "sparsifier/collect_statistics",
                         "Should we collect statistics about the original "
                         "and the sparsified cuts?",
                         &collect_statistics_, true, false, nullptr, nullptr),
        SCIP_OKAY);
    CHECK_EQ(SCIPaddCharParam(scip, "sparsifier/efficacythreshold",
                              "How to determine the efficacy threshold: "
                              "(s)election, (c)urrent, (n)one. All use "
                              "separating/minefficacy as the absolute minimum.",
                              &efficacy_threshold_method_, false, 's', "scn",
                              nullptr, nullptr),
             SCIP_OKAY);
    CHECK_EQ(
        SCIPaddIntParam(
            scip, "sparsifier/maxselectednnz",
            "Maximum number of nonzeros in all selected cuts. (-1: unlimited)",
            &max_selected_nnz_, false, -1, -1, INT_MAX, nullptr, nullptr),
        SCIP_OKAY);
    CHECK_EQ(
        SCIPaddRealParam(
            scip, "sparsifier/maxdensity",
            "Upper limit on density of selected cuts, after sparsification.",
            &max_density_, false, 1.0, 0.0, 1.0, nullptr, nullptr),
        SCIP_OKAY);
  }

  SCIP_DECL_CUTSELINIT(scip_init) final {
    SCIPcreateRandom(scip, &hybrid_randnumgen_, kHybridRandSeed, true);
    return SCIP_OKAY;
  }

  SCIP_DECL_CUTSELEXIT(scip_exit) final {
    SCIPfreeRandom(scip, &hybrid_randnumgen_);
    return SCIP_OKAY;
  }

  SCIP_DECL_CUTSELSELECT(scip_select) final {
    // By sorting the cuts by index, we make the statistics easier to parse.
    std::sort(cuts, cuts + ncuts, [](SCIP_ROW* r1, SCIP_ROW* r2) {
      return SCIProwGetIndex(r1) < SCIProwGetIndex(r2);
    });

    // We can select at most N cuts. We therefore use the efficacy of the Nth
    // most efficacious as the efficacy target of the sparsified cuts.
    // TODO: Try using DCD instead.
    double min_efficacy;
    SCIP_CALL(SCIPgetRealParam(scip, "separating/minefficacy", &min_efficacy));
    const double selection_based_efficacy_target = [&]() {
      std::vector<double> efficacies(ncuts);
      std::transform(cuts, cuts + ncuts, efficacies.begin(),
                     [scip](SCIP_ROW* cut) {
                       return SCIPgetCutEfficacy(scip, nullptr, cut);
                     });
      if (efficacies.size() <= maxnselectedcuts) {
        return *std::min_element(efficacies.begin(), efficacies.end());
      }
      std::partial_sort(efficacies.begin(),
                        efficacies.begin() + maxnselectedcuts, efficacies.end(),
                        std::greater<>{});
      return *(efficacies.begin() + maxnselectedcuts);
    }();

    for (int i = 0; i < ncuts; ++i) {
      SCIP_ROW* cut = cuts[i];
      const int index = SCIProwGetIndex(cut);

      // Since we pass the sparsified cuts directly to the LP, we don't expect
      // them to show up again.
      DCHECK(!sparsified_cuts_.contains(index))
          << index << " is already a sparsified cut.";

      SCIP_ROW* sparsified = nullptr;
      if (call_sparsifier_) {
        double target_efficacy;
        if (efficacy_threshold_method_ == 's') {
          target_efficacy = selection_based_efficacy_target;
        } else if (efficacy_threshold_method_ == 'c') {
          target_efficacy = SCIPgetCutEfficacy(scip, nullptr, cut);
        } else if (efficacy_threshold_method_ == 'n') {
          target_efficacy = min_efficacy;
        } else {
          LOG(FATAL) << "Unknown efficacy threshold method: "
                     << efficacy_threshold_method_;
        }
        target_efficacy = std::max(target_efficacy, min_efficacy);
        SCIP_CALL(
            sparsifier_->Sparsify(scip, cut, &sparsified, target_efficacy));
      }
      if (sparsified == nullptr) {
        // If the cut wasn't sparsified, we use the original cut to simply the
        // remaining code below.
        sparsified = cut;
      } else {
        const int sparse_index = SCIProwGetIndex(sparsified);
        DCHECK_GT(sparse_index, index);
        sparsified_cuts_.insert(sparse_index);
      }

      if (replace_original_) cuts[i] = sparsified;

      if (collect_statistics_) {
        auto clock_restarter = MeassureTime();
        CutStatistics original_stat(scip, cut);
        CutStatistics sparsified_stat(scip, sparsified);
        sparsified_stat.sparsification = sparsifier_->GetNameSuffix();
        table_->AddStats(original_stat, sparsified_stat);
      }
    }

    // Reject cuts that are still too dense.
    ncuts = std::distance(
        cuts, std::stable_partition(
                  cuts, cuts + ncuts,
                  [nvars = SCIPgetNVars(scip),
                   max_density = max_density_](SCIP_ROW* cut) {
                    return SCIProwGetNNonz(cut) / static_cast<double>(nvars) <=
                           max_density;
                  }));
    if (ncuts == 0) {
      *nselectedcuts = 0;
      return SCIP_OKAY;
    }

    // We now call the default (hybrid) cut selector for the actual
    // selection. The parameter values are taken to be the same ones used in
    // `cutsel_hybrid.c`.
    double hybrid_dircutoffdistweight;
    SCIP_CALL(SCIPgetRealParam(scip, "cutselection/hybrid/dircutoffdistweight",
                               &hybrid_dircutoffdistweight));
    double hybrid_efficacyweight;
    SCIP_CALL(SCIPgetRealParam(scip, "cutselection/hybrid/efficacyweight",
                               &hybrid_efficacyweight));
    double hybrid_objparalweight;
    SCIP_CALL(SCIPgetRealParam(scip, "cutselection/hybrid/objparalweight",
                               &hybrid_objparalweight));
    double hybrid_intsupportweight;
    SCIP_CALL(SCIPgetRealParam(scip, "cutselection/hybrid/intsupportweight",
                               &hybrid_intsupportweight));
    double hybrid_minortho;
    if (root) {
      SCIP_CALL(SCIPgetRealParam(scip, "cutselection/hybrid/minorthoroot",
                                 &hybrid_minortho));
    } else {
      SCIP_CALL(SCIPgetRealParam(scip, "cutselection/hybrid/minortho",
                                 &hybrid_minortho));
    }
    const double hybrid_maxparall = 1.0 - hybrid_minortho;
    const double hybrid_goodmaxparall = std::max(0.5, 1.0 - hybrid_minortho);

    int hybrid_nselectedcuts;
    SCIP_CALL(SCIPselectCutsHybrid(
        scip, cuts, forcedcuts, hybrid_randnumgen_, kHybridGoodScore,
        kHybridBadScore, hybrid_goodmaxparall, hybrid_maxparall,
        hybrid_dircutoffdistweight, hybrid_efficacyweight,
        hybrid_objparalweight, hybrid_intsupportweight, ncuts, nforcedcuts,
        maxnselectedcuts, &hybrid_nselectedcuts));

    // Reject cuts that would bring us above the nnz limit.
    if (max_selected_nnz_ >= 0) {
      const std::vector<SCIP_ROW*> cuts_copy(cuts, cuts + ncuts);
      SCIP_ROW** front = cuts;
      SCIP_ROW** back = cuts + ncuts;
      *nselectedcuts = 0;
      for (int i = 0; i < hybrid_nselectedcuts; ++i) {
        SCIP_ROW* cut = cuts_copy[i];
        const int nnz = SCIProwGetNNonz(cut);
        if (nnz + total_selected_nnz_ <= max_selected_nnz_) {
          *nselectedcuts += 1;
          total_selected_nnz_ += nnz;
          *front = cut;
          front += 1;
        } else {
          back -= 1;
          *back = cut;
        }
      }
      DCHECK_EQ(std::distance(front, back), ncuts - hybrid_nselectedcuts);
      std::copy(cuts_copy.begin() + hybrid_nselectedcuts, cuts_copy.end(),
                front);
    } else {
      *nselectedcuts = hybrid_nselectedcuts;
    }

    // Store nnz statistics.
    for (int i = 0; i < *nselectedcuts; ++i) nnz_table_->AddCut(cuts[i]);

    *result = SCIP_SUCCESS;
    return SCIP_OKAY;
  }

 private:
  // Parameters
  SCIP_Bool call_sparsifier_;
  SCIP_Bool replace_original_;
  SCIP_Bool collect_statistics_;
  double max_density_;
  int max_selected_nnz_;
  char efficacy_threshold_method_;

  int total_selected_nnz_ = 0;

  std::unique_ptr<SparsifierInterface> sparsifier_;
  CutByOriginTable* table_;
  NonzeroTable* nnz_table_;
  absl::flat_hash_set<int> sparsified_cuts_;

  static constexpr int kHybridRandSeed = 0x5EED;
  static constexpr double kHybridGoodScore = 0.9;
  static constexpr double kHybridBadScore = 0.0;

  SCIP_RANDNUMGEN* hybrid_randnumgen_;
};

SCIP_RETCODE RunExperiment() {
  const std::string config_file = absl::GetFlag(FLAGS_config_file);
  const std::string problem_name = absl::GetFlag(FLAGS_problem_name);
  const std::string problem_file = absl::GetFlag(FLAGS_problem_file);
  const std::string solution_file = absl::GetFlag(FLAGS_solution_file);
  const std::string result_dir = absl::GetFlag(FLAGS_result_dir);
  const int seed = absl::GetFlag(FLAGS_seed);
  const int max_selected_nnz = absl::GetFlag(FLAGS_max_selected_nnz);
  const bool default_config = absl::GetFlag(FLAGS_default_config);

  // SCIP sometimes doesn't print the final linebreak.
  auto linebreak = absl::MakeCleanup([]() { std::cout << std::endl; });

  SCIP* scip = nullptr;
  SCIP_CALL(SCIPcreate(&scip));
  SCIP_CALL(SCIPincludeDefaultPlugins(scip));
  SCIPprintVersion(scip, nullptr);
  SCIPprintExternalCodes(scip, nullptr);
  SCIPprintBuildOptions(scip, nullptr);
  SCIPdebugMessagePrint(scip, "\n");

  std::unique_ptr<CutByOriginTable> stats_table;
  std::unique_ptr<NonzeroTable> nnz_table;
  std::unique_ptr<CutStatisticsCollector> stats_collector;
  if (!default_config) {
    stats_table = std::make_unique<CutByOriginTable>(
        scip, absl::StrCat(result_dir, "/original_cut_stats.csv"),
        absl::StrCat(result_dir, "/sparsified_cut_stats.csv"));
    SCIP_CALL(SCIPincludeObjTable(scip, stats_table.get(), false));

    nnz_table = std::make_unique<NonzeroTable>(scip);
    SCIP_CALL(SCIPincludeObjTable(scip, nnz_table.get(), false));

    auto sparsifier = std::make_unique<ActivationSparsifier>(scip);
    stats_collector = std::make_unique<CutStatisticsCollector>(
        scip, std::move(sparsifier), stats_table.get(), nnz_table.get());
    SCIP_CALL(SCIPincludeObjCutsel(scip, stats_collector.get(), false));
  }

  SCIP_CALL(SCIPreadParams(scip, config_file.c_str()));
  SCIP_CALL(SCIPsetIntParam(scip, "randomization/randomseedshift", seed));
  if (!default_config) {
    SCIP_CALL(
        SCIPsetIntParam(scip, "sparsifier/maxselectednnz", max_selected_nnz));
  }
  SCIP_CALL(SCIPreadProb(scip, problem_file.c_str(), nullptr));
  SCIP_CALL(SCIPreadSol(scip, solution_file.c_str()));
  SCIP_CALL(SCIPsolve(scip));
  SCIP_CALL(SCIPprintStatistics(scip, nullptr));
  SCIP_CALL(SCIPwriteParams(
      scip, absl::StrCat(result_dir, "/settings.set").c_str(), true, false));
  SCIP_CALL(SCIPfree(&scip));

  return SCIP_OKAY;
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  CHECK(!absl::GetFlag(FLAGS_config_file).empty());
  CHECK(!absl::GetFlag(FLAGS_problem_name).empty());
  CHECK(!absl::GetFlag(FLAGS_problem_file).empty());
  CHECK(!absl::GetFlag(FLAGS_solution_file).empty());
  CHECK(!absl::GetFlag(FLAGS_result_dir).empty());
  CHECK(IsDirectory(absl::GetFlag(FLAGS_result_dir)))
      << absl::GetFlag(FLAGS_result_dir)
      << " doesn't name an existing directory.";
  CHECK_NE(absl::GetFlag(FLAGS_seed), -1) << "Seed must be provided.";
  SCIP_CALL(RunExperiment());
  return 0;
}
