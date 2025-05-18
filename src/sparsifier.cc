#include "sparsifier.h"

#include <scip/pub_lp.h>
#include <scip/scip_lp.h>
#include <scip/scip_numerics.h>
#include <scip/type_retcode.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <queue>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "statistics.h"

namespace {

struct Nonzero {
  SCIP_COL* col;
  double coefficient;
  absl::optional<double> bound;

  Nonzero(SCIP_COL* col, double coefficient, absl::optional<double> bound)
      : col{col}, coefficient{coefficient}, bound{bound} {}
};

SCIP_Real GetUpperBound(SCIP_COL* col) {
  return SCIPvarGetUbGlobal(SCIPcolGetVar(col));
}

bool HasUpperBound(SCIP* scip, SCIP_COL* col) {
  return !SCIPisInfinity(scip, GetUpperBound(col));
}

SCIP_Real GetLowerBound(SCIP_COL* col) {
  return SCIPvarGetLbGlobal(SCIPcolGetVar(col));
}

bool HasLowerBound(SCIP* scip, SCIP_COL* col) {
  return !SCIPisInfinity(scip, -GetLowerBound(col));
}

absl::flat_hash_map<int, double> ExtractNonzeroMaps(SCIP* scip, SCIP_ROW* row) {
  absl::flat_hash_map<int, double> coefficients;
  SCIP_COL** cols = SCIProwGetCols(row);
  double* vals = SCIProwGetVals(row);
  const double factor = SCIPisInfinity(scip, SCIProwGetRhs(row)) ? -1 : 1;
  for (int i = 0; i < SCIProwGetNNonz(row); ++i) {
    const int index = SCIPcolGetIndex(cols[i]);
    DCHECK(!coefficients.contains(index))
        << "Row " << SCIProwGetName(row) << " (index: " << SCIProwGetIndex(row)
        << ") contains multiple coefficients for column " << index << ".";
    coefficients[index] = factor * vals[i];
  }
  return coefficients;
}

std::vector<double> ExtractCoefficients(SCIP* scip, SCIP_ROW* row) {
  std::vector<double> coefficients(SCIProwGetVals(row),
                                   SCIProwGetVals(row) + SCIProwGetNNonz(row));
  if (SCIPisInfinity(scip, SCIProwGetRhs(row))) {
    std::transform(coefficients.begin(), coefficients.end(),
                   coefficients.begin(), [](double d) { return -d; });
  }
  return coefficients;
}

double ExtractRhs(SCIP* scip, SCIP_ROW* row) {
  const double lhs = SCIProwGetLhs(row);
  const double rhs = SCIProwGetRhs(row);
  const double constant = SCIProwGetConstant(row);
  const double side_value =
      SCIPisInfinity(scip, rhs) ? -(lhs - constant) : (rhs - constant);
  DCHECK(!SCIPisInfinity(scip, side_value));
  DCHECK(!SCIPisInfinity(scip, -side_value));
  return side_value;
}

std::vector<Nonzero> ExtractNonzeros(SCIP* scip, SCIP_ROW* row) {
  const double coefficient_factor =
      SCIPisInfinity(scip, SCIProwGetRhs(row)) ? -1 : 1;
  const int nnz = SCIProwGetNNonz(row);
  std::vector<Nonzero> res;
  res.reserve(nnz);
  for (int i = 0; i < nnz; ++i) {
    SCIP_COL* col = SCIProwGetCols(row)[i];
    const double coefficient = coefficient_factor * SCIProwGetVals(row)[i];
    absl::optional<double> bound = absl::nullopt;
    if (coefficient >= 0 && HasLowerBound(scip, col)) {
      bound = GetLowerBound(col);
    } else if (coefficient <= 0 && HasUpperBound(scip, col)) {
      bound = GetUpperBound(col);
    }
    res.emplace_back(col, coefficient, bound);
  }
  return res;
}

std::string ExtractRowOriginName(SCIP* scip, SCIP_ROW* row) {
  SCIP_SEPA* sepa = SCIProwGetOriginSepa(row);
  if (sepa != nullptr) return SCIPsepaGetName(sepa);
  SCIP_CONS* cons = SCIProwGetOriginCons(row);
  if (cons != nullptr) return SCIPconsGetName(cons);
  SCIP_CONSHDLR* conshdlr = SCIProwGetOriginConshdlr(row);
  if (conshdlr != nullptr) return SCIPconshdlrGetName(conshdlr);
  return absl::StrFormat("unknown [%d]", SCIProwGetOrigintype(row));
}

SCIP_Retcode CreateAndPopulateRow(
    SCIP* scip, SCIP_ROW* original, SCIP_ROW** sparsified,
    const std::vector<std::pair<SCIP_COL*, double>>& sparsified_coefficients,
    double rhs) {
  DCHECK_NE(original, nullptr);
  DCHECK_NE(sparsified, nullptr);
  DCHECK_EQ(*sparsified, nullptr);

  const double inf = SCIPinfinity(scip);
  const bool local = SCIProwIsLocal(original);
  const bool modifiable = SCIProwIsModifiable(original);
  const bool removable = SCIProwIsRemovable(original);
  const char* name = SCIProwGetName(original);
  switch (SCIProwGetOrigintype(original)) {
    case SCIP_ROWORIGINTYPE_SEPA: {
      SCIP_SEPA* sepa = SCIProwGetOriginSepa(original);
      DCHECK_NE(sepa, nullptr);
      SCIP_CALL(SCIPcreateEmptyRowSepa(scip, sparsified, sepa, name, -inf, inf,
                                       local, modifiable, removable));
      break;
    }
    case SCIP_ROWORIGINTYPE_CONS: {
      SCIP_CONS* cons = SCIProwGetOriginCons(original);
      DCHECK_NE(cons, nullptr);
      SCIP_CALL(SCIPcreateEmptyRowCons(scip, sparsified, cons, name, -inf, inf,
                                       local, modifiable, removable));
      break;
    }
    case SCIP_ROWORIGINTYPE_CONSHDLR: {
      SCIP_CONSHDLR* conshdlr = SCIProwGetOriginConshdlr(original);
      DCHECK_NE(conshdlr, nullptr);
      SCIP_CALL(SCIPcreateEmptyRowConshdlr(scip, sparsified, conshdlr, name,
                                           -inf, inf, local, modifiable,
                                           removable));
      break;
    }
    case SCIP_ROWORIGINTYPE_REOPT:
    case SCIP_ROWORIGINTYPE_UNSPEC:
      SCIP_CALL(SCIPcreateEmptyRowUnspec(scip, sparsified, name, -inf, inf,
                                         local, modifiable, removable));
      break;
  }

  SCIP_CALL(SCIPcacheRowExtensions(scip, *sparsified));
  SCIPchgRowRhs(scip, *sparsified, rhs);
  for (auto [column, coefficient] : sparsified_coefficients) {
    DCHECK_NE(column, nullptr);
    SCIP_CALL(
        SCIPaddVarToRow(scip, *sparsified, SCIPcolGetVar(column), coefficient));
  }
  SCIP_CALL(SCIPflushRowExtensions(scip, *sparsified));
  return SCIP_OKAY;
}

std::string PrintCutComparison(SCIP* scip, SCIP_ROW* row1, SCIP_ROW* row2) {
  SCIP_SOL* opt_sol = SCIPgetBestSol(scip);
  SCIPdebugMessagePrint(scip, "Comparing rows %d and %d\n",
                        SCIProwGetIndex(row1), SCIProwGetIndex(row2));
  SCIPdebugMessagePrint(scip, "Names: %s vs %s\n", SCIProwGetName(row1),
                        SCIProwGetName(row2));
  SCIPdebugMessagePrint(scip, "Origin: %s vs %s\n",
                        ExtractRowOriginName(scip, row1).c_str(),
                        ExtractRowOriginName(scip, row2).c_str());
  SCIPdebugMessagePrint(scip, "Efficacy: %lf vs %lf\n",
                        SCIPgetCutEfficacy(scip, nullptr, row1),
                        SCIPgetCutEfficacy(scip, nullptr, row2));
  SCIPdebugMessagePrint(scip, "Activity: %lf vs %lf\n",
                        SCIPgetRowActivity(scip, row1),
                        SCIPgetRowActivity(scip, row2));
  SCIPdebugMessagePrint(scip, "Norm: %lf vs %lf\n", SCIProwGetNorm(row1),
                        SCIProwGetNorm(row2));
  SCIPdebugMessagePrint(scip, "Feasibility: %lf vs %lf\n",
                        SCIPgetRowFeasibility(scip, row1),
                        SCIPgetRowFeasibility(scip, row2));
  if (opt_sol != nullptr) {
    SCIPdebugMessagePrint(scip, "Opt activity: %lf vs %lf\n",
                          SCIPgetRowSolActivity(scip, row1, opt_sol),
                          SCIPgetRowSolActivity(scip, row2, opt_sol));
    SCIPdebugMessagePrint(scip, "Opt efficacy: %lf vs %lf\n",
                          SCIPgetCutEfficacy(scip, opt_sol, row1),
                          SCIPgetCutEfficacy(scip, opt_sol, row2));
    SCIPdebugMessagePrint(scip, "Opt feasibility: %lf vs %lf\n",
                          SCIPgetRowSolFeasibility(scip, row1, opt_sol),
                          SCIPgetRowSolFeasibility(scip, row2, opt_sol));
    SCIPdebugMessagePrint(scip, "Directed cutoff distance: %lf vs %lf\n",
                          SCIPgetCutLPSolCutoffDistance(scip, opt_sol, row1),
                          SCIPgetCutLPSolCutoffDistance(scip, opt_sol, row2));
  }
  SCIPdebugMessagePrint(scip, "RHS: %lf vs %lf\n", SCIProwGetRhs(row1),
                        SCIProwGetRhs(row2));
  SCIPdebugMessagePrint(scip, "LHS: %lf vs %lf\n", SCIProwGetLhs(row1),
                        SCIProwGetLhs(row2));
  SCIPdebugMessagePrint(scip, "Constant: %lf vs %lf\n",
                        SCIProwGetConstant(row1), SCIProwGetConstant(row2));

  absl::flat_hash_map<int, double> a1;
  absl::flat_hash_map<int, double> a2;
  absl::flat_hash_map<int, double> lower_bounds;
  absl::flat_hash_map<int, double> upper_bounds;
  absl::flat_hash_map<int, double> prim_val;
  absl::flat_hash_map<int, double> opt_val;
  std::set<int> all_vars;

  for (int i = 0; i < SCIProwGetNNonz(row1); ++i) {
    SCIP_COL* col = SCIProwGetCols(row1)[i];
    const int xi = SCIPcolGetIndex(col);
    lower_bounds[xi] = SCIPvarGetLbGlobal(SCIPcolGetVar(col));
    upper_bounds[xi] = SCIPvarGetUbGlobal(SCIPcolGetVar(col));
    prim_val[xi] = SCIPcolGetPrimsol(col);
    if (opt_sol != nullptr) {
      opt_val[xi] = SCIPgetSolVal(scip, opt_sol, SCIPcolGetVar(col));
    }
    all_vars.insert(xi);
    a1[xi] = SCIProwGetVals(row1)[i];
  }
  for (int i = 0; i < SCIProwGetNNonz(row2); ++i) {
    SCIP_COL* col = SCIProwGetCols(row2)[i];
    const int xi = SCIPcolGetIndex(col);
    lower_bounds[xi] = SCIPvarGetLbGlobal(SCIPcolGetVar(col));
    upper_bounds[xi] = SCIPvarGetUbGlobal(SCIPcolGetVar(col));
    prim_val[xi] = SCIPcolGetPrimsol(col);
    if (opt_sol != nullptr) {
      opt_val[xi] = SCIPgetSolVal(scip, opt_sol, SCIPcolGetVar(col));
    }
    all_vars.insert(xi);
    a2[xi] = SCIProwGetVals(row2)[i];
  }
  for (int xi : all_vars) {
    if (a1[xi] == -a2[xi]) {
      SCIPdebugMessagePrint(scip, "SIGN     ");
    } else if (a1[xi] != a2[xi]) {
      SCIPdebugMessagePrint(scip, "MISMATCH ");
    } else {
      SCIPdebugMessagePrint(scip, "         ");
    }
    SCIPdebugMessagePrint(scip,
                          "% lf * x%4i vs % lf * x%4i [lower_bound=% lf, "
                          "upper_bound=% lf, lpval=% lf",
                          a1[xi], xi, a2[xi], xi, lower_bounds[xi],
                          upper_bounds[xi], prim_val[xi]);
    if (opt_sol != nullptr) {
      SCIPdebugMessagePrint(scip, ", optval=% lf", opt_val[xi]);
    }
    SCIPdebugMessagePrint(scip, "]\n");
  }

  return "";
}

void VerifySparsifiedCut(SCIP* scip, SCIP_ROW* sparsified, SCIP_ROW* original) {
  const absl::flat_hash_map<int, double> original_coefficients =
      ExtractNonzeroMaps(scip, original);
  const double original_rhs = ExtractRhs(scip, original);
  const absl::flat_hash_map<int, double> sparsified_coefficients =
      ExtractNonzeroMaps(scip, sparsified);
  const double sparsified_rhs = ExtractRhs(scip, sparsified);

  // Even if the cuts are the same, the sparsified cut must be a deep copy and
  // hence have its own unique index.
  CHECK_NE(SCIProwGetIndex(sparsified), SCIProwGetIndex(original));

  for (const auto [index, value] : sparsified_coefficients) {
    // No explicit zero coefficients.
    CHECK_NE(value, 0.0) << PrintCutComparison(scip, sparsified, original);

    // No variables not in original.
    CHECK(original_coefficients.contains(index))
        << PrintCutComparison(scip, original, sparsified);

    // Retained variables should have unchanged coefficient
    CHECK_EQ(value, sparsified_coefficients.at(index))
        << PrintCutComparison(scip, sparsified, original);
  }

  double sparsified_rhs_limit = original_rhs;
  for (int i = 0; i < SCIProwGetNNonz(original); ++i) {
    SCIP_COL* col = SCIProwGetCols(original)[i];
    const int index = SCIPcolGetIndex(col);

    // The unchanged coefficients are already checked above.
    if (sparsified_coefficients.contains(SCIPcolGetIndex(col))) continue;

    if (original_coefficients.at(index) >= 0) {
      CHECK(HasLowerBound(scip, col))
          << PrintCutComparison(scip, sparsified, original);
      sparsified_rhs_limit -=
          GetLowerBound(col) * original_coefficients.at(index);
    } else {
      CHECK(HasUpperBound(scip, col))
          << PrintCutComparison(scip, sparsified, original);
      sparsified_rhs_limit -=
          GetUpperBound(col) * original_coefficients.at(index);
    }
  }
  // The bound substitution gives the lowest possible value for the right hand
  // side that guarantees validity.
  CHECK_GE(sparsified_rhs + 1e-6, sparsified_rhs_limit)
      << PrintCutComparison(scip, sparsified, original);

  // Cuts should be efficacious.
  const double original_efficacy = SCIPgetCutEfficacy(scip, nullptr, original);
  const double sparsified_efficacy =
      SCIPgetCutEfficacy(scip, nullptr, sparsified);
  // TODO Handle inefficacious cuts in a nicer way
  // CHECK_GT(original_efficacy, 0.0)
  //     << PrintCutComparison(scip, sparsified, original);
  // CHECK_GT(sparsified_efficacy, 0.0)
  //     << PrintCutComparison(scip, sparsified, original);
  if (sparsified_coefficients.size() == original_coefficients.size()) {
    CHECK(abs(sparsified_efficacy - original_efficacy) < 1e-4)
        << PrintCutComparison(scip, sparsified, original);
  }
}

SCIP_Retcode GetGlobalBounds(SCIP_ROW* cut, std::vector<double>& bounds) {
  DCHECK(bounds.empty());
  SCIP_COL** columns = SCIProwGetCols(cut);
  double* coefficients = SCIProwGetVals(cut);
  const int nnz = SCIProwGetNNonz(cut);
  bounds.reserve(nnz);
  for (int i = 0; i < nnz; ++i) {
    bounds.push_back(coefficients[i] >= 0 ? GetLowerBound(columns[i])
                                          : GetUpperBound(columns[i]));
  }
  return SCIP_OKAY;
}

}  // namespace

SCIP_Retcode CopyRow(SCIP* scip, SCIP_ROW* original, SCIP_ROW** copy) {
  const double rhs = ExtractRhs(scip, original);
  const double factor = SCIPisInfinity(scip, SCIProwGetRhs(original)) ? -1 : 1;
  const int nnz = SCIProwGetNNonz(original);
  std::vector<std::pair<SCIP_COL*, double>> coefficients;
  coefficients.reserve(nnz);
  for (int i = 0; i < nnz; ++i) {
    coefficients.emplace_back(SCIProwGetCols(original)[i],
                              factor * SCIProwGetVals(original)[i]);
  }
  SCIP_CALL(CreateAndPopulateRow(scip, original, copy, coefficients, rhs));
  return SCIP_OKAY;
}

SCIP_Retcode ActivationSparsifier::Sparsify(SCIP* scip, SCIP_ROW* original,
                                            SCIP_ROW** sparsified,
                                            double min_efficacy) {
  DCHECK_NE(original, nullptr);
  DCHECK_NE(sparsified, nullptr);
  DCHECK_EQ(*sparsified, nullptr);
  DCHECK_GT(min_efficacy, 0.0);

  const int nnz = SCIProwGetNNonz(original);
  const bool has_rhs = !SCIPisInfinity(scip, SCIProwGetRhs(original));
  const bool has_lhs = !SCIPisInfinity(scip, -SCIProwGetLhs(original));
  DCHECK(has_rhs || has_lhs);
  const bool ignore_local = SCIProwIsLocal(original) && !sparsify_local_;
  const bool two_sided = has_rhs && has_lhs;
  const bool already_below_min =
      SCIPgetCutEfficacy(scip, nullptr, original) < min_efficacy;
  const bool already_sparse =
      nnz / static_cast<double>(SCIPgetNVars(scip)) < min_density_;
  const bool can_sparsify =
      !ignore_local && !two_sided && !already_below_min && !already_sparse;
  if (!can_sparsify) return SCIP_OKAY;

  SCIP_COL** columns = SCIProwGetCols(original);
  double side_value = ExtractRhs(scip, original);

  // This is where we store the nonzeros for the sparsified cut.
  std::vector<std::pair<SCIP_COL*, double>> sparsified_coefficients;
  sparsified_coefficients.reserve(nnz);

  std::vector<Nonzero> original_nonzeros = ExtractNonzeros(scip, original);

  // Tentatively remove all bounded variables.
  double included_activation = 0.0;
  double included_norm = 0.0;
  std::vector<double> feasibility_impact(nnz, 0.0);
  std::vector<int> removed_variables;
  removed_variables.reserve(nnz);
  for (int i = 0; i < nnz; ++i) {
    const Nonzero& nonzero = original_nonzeros[i];
    const double primval = SCIPcolGetPrimsol(nonzero.col);
    if (nonzero.bound.has_value()) {
      if (nonzero.coefficient >= 0) {
        const double lb = *nonzero.bound;
        feasibility_impact[i] =
            nonzero.coefficient * std::max(primval - lb, 0.0);
      } else {
        const double ub = *nonzero.bound;
        feasibility_impact[i] =
            nonzero.coefficient * std::min(primval - ub, 0.0);
      }
    }
    // Note that if there was no relevant bound, the feasibility impact is 0
    // from the initialization. This is fine, since we won't use it unless the
    // nonzero can be removed.
    DCHECK_GE(feasibility_impact[i], 0.0);

    const bool can_be_removed =
        nonzero.bound.has_value() &&
        (reduce_activation_ || feasibility_impact[i] == 0);
    if (can_be_removed) {
      removed_variables.push_back(i);
      side_value -= nonzero.coefficient * nonzero.bound.value();
    } else {
      sparsified_coefficients.emplace_back(nonzero.col, nonzero.coefficient);
      included_activation += nonzero.coefficient * primval;
      included_norm += nonzero.coefficient * nonzero.coefficient;
    }
  }

  // Reintroduce nonzeros in order of contributed activation.
  std::sort(removed_variables.begin(), removed_variables.end(),
            [&feasibility_impact](int i1, int i2) {
              return feasibility_impact[i1] > feasibility_impact[i2];
            });
  int num_reintroduced;
  for (num_reintroduced = 0; num_reintroduced < removed_variables.size();
       ++num_reintroduced) {
    int var_index = removed_variables[num_reintroduced];
    const Nonzero& nonzero = original_nonzeros[var_index];
    const double primval = SCIPcolGetPrimsol(nonzero.col);
    const double feasibility = included_activation - side_value;
    if (feasibility >= 0 &&
        feasibility / std::sqrt(included_norm) >= min_efficacy) {
      break;
    }
    sparsified_coefficients.emplace_back(nonzero.col, nonzero.coefficient);
    // Note that bounds is the appropriate upper/lower bound depending on the
    // coefficient sign.
    DCHECK(nonzero.bound.has_value());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    side_value += nonzero.coefficient * nonzero.bound.value();
    included_activation += nonzero.coefficient * primval;
    included_norm += nonzero.coefficient * nonzero.coefficient;
  }

  // If we couldn't remove any nonzeros, we don't need to copy the cut.
  if (sparsified_coefficients.size() == nnz) return SCIP_OKAY;

  SCIP_CALL(CreateAndPopulateRow(scip, original, sparsified,
                                 sparsified_coefficients, side_value));

  // Sanity checks.
  DCHECK_GE(SCIPgetCutEfficacy(scip, nullptr, original) + 1e-6, min_efficacy);
  DCHECK_GE(SCIPgetCutEfficacy(scip, nullptr, *sparsified) + 1e-6,
            min_efficacy);

#ifndef NDEBUG
  VerifySparsifiedCut(scip, *sparsified, original);
#endif  // NDEBUG

  return SCIP_OKAY;
}
