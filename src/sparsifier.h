#pragma once

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "objscip/objscip.h"

SCIP_Retcode CopyRow(SCIP* scip, SCIP_ROW* original, SCIP_ROW** copy);

class SparsifierInterface {
 public:
  virtual ~SparsifierInterface() = default;

  virtual SCIP_Retcode Sparsify(SCIP* scip, SCIP_ROW* original,
                                SCIP_ROW** sparsified, double min_efficacy) = 0;

  virtual std::string GetNameSuffix() const = 0;
};

class ActivationSparsifier : public SparsifierInterface {
 public:
  explicit ActivationSparsifier(SCIP* scip) {
    CHECK_EQ(SCIPaddBoolParam(scip, "sparsifier/activation/sparsify_local",
                              "Should local cuts be sparsified?",
                              &sparsify_local_, false, false, nullptr, nullptr),
             SCIP_OKAY);
    CHECK_EQ(SCIPaddRealParam(
                 scip, "sparsifier/activation/min_density",
                 "Only cuts with at least the given density are sparsified.",
                 &min_density_, false, 0.0, 0.0, 1.0, nullptr, nullptr),
             SCIP_OKAY);
    CHECK_EQ(SCIPaddIntParam(scip, "sparsifier/activation/bounditerations",
                             "Limits the number of LP iterations per variable "
                             "to find the local bound (-1: unlimited).",
                             &bound_dive_iteration_limit_, false, 5, -1,
                             INT_MAX, nullptr, nullptr),
             SCIP_OKAY);
    CHECK_EQ(SCIPaddBoolParam(
                 scip, "sparsifier/activation/reduceactivation",
                 "Allow removing nonzeros which reduce the cut activation.",
                 &reduce_activation_, false, true, nullptr, nullptr),
             SCIP_OKAY);
  }

  SCIP_Retcode Sparsify(SCIP* scip, SCIP_ROW* original, SCIP_ROW** sparsified,
                        double min_efficacy) final;

  std::string GetNameSuffix() const final { return "by_activation"; }

 private:
  SCIP_Bool sparsify_local_;
  SCIP_Real min_density_;
  SCIP_Bool reduce_activation_;
  int bound_dive_iteration_limit_;
};
