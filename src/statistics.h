#include <fstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "objscip/objscip.h"

template <typename T>
void DumpStatisticsToFile(SCIP* scip, const std::vector<T>& v,
                          const std::string& filename) {
  std::fstream fs(filename, std::ios_base::out);
  CHECK(fs.good()) << "Failed to open file" << filename;
  fs << T::CSVHeader() << "\n";
  for (const T& row : v) fs << row.ToCSVFormat() << "\n";
}