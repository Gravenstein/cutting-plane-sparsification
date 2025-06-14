include(CMakeParseArguments)
include(GNUInstallDirs)

set(ABSL_INTERNAL_DLL_FILES
  "algorithm/algorithm.h"
  "algorithm/container.h"
  "base/attributes.h"
  "base/call_once.h"
  "base/casts.h"
  "base/config.h"
  "base/const_init.h"
  "base/dynamic_annotations.h"
  "base/internal/atomic_hook.h"
  "base/internal/cycleclock.cc"
  "base/internal/cycleclock.h"
  "base/internal/cycleclock_config.h"
  "base/internal/direct_mmap.h"
  "base/internal/dynamic_annotations.h"
  "base/internal/endian.h"
  "base/internal/errno_saver.h"
  "base/internal/fast_type_id.h"
  "base/internal/hide_ptr.h"
  "base/internal/identity.h"
  "base/internal/invoke.h"
  "base/internal/inline_variable.h"
  "base/internal/low_level_alloc.cc"
  "base/internal/low_level_alloc.h"
  "base/internal/low_level_scheduling.h"
  "base/internal/per_thread_tls.h"
  "base/internal/prefetch.h"
  "base/internal/pretty_function.h"
  "base/internal/raw_logging.cc"
  "base/internal/raw_logging.h"
  "base/internal/scheduling_mode.h"
  "base/internal/scoped_set_env.cc"
  "base/internal/scoped_set_env.h"
  "base/internal/strerror.h"
  "base/internal/strerror.cc"
  "base/internal/spinlock.cc"
  "base/internal/spinlock.h"
  "base/internal/spinlock_wait.cc"
  "base/internal/spinlock_wait.h"
  "base/internal/sysinfo.cc"
  "base/internal/sysinfo.h"
  "base/internal/thread_annotations.h"
  "base/internal/thread_identity.cc"
  "base/internal/thread_identity.h"
  "base/internal/throw_delegate.cc"
  "base/internal/throw_delegate.h"
  "base/internal/tsan_mutex_interface.h"
  "base/internal/unaligned_access.h"
  "base/internal/unscaledcycleclock.cc"
  "base/internal/unscaledcycleclock.h"
  "base/internal/unscaledcycleclock_config.h"
  "base/log_severity.cc"
  "base/log_severity.h"
  "base/macros.h"
  "base/optimization.h"
  "base/options.h"
  "base/policy_checks.h"
  "base/port.h"
  "base/thread_annotations.h"
  "cleanup/cleanup.h"
  "cleanup/internal/cleanup.h"
  "container/btree_map.h"
  "container/btree_set.h"
  "container/fixed_array.h"
  "container/flat_hash_map.h"
  "container/flat_hash_set.h"
  "container/inlined_vector.h"
  "container/internal/btree.h"
  "container/internal/btree_container.h"
  "container/internal/common.h"
  "container/internal/common_policy_traits.h"
  "container/internal/compressed_tuple.h"
  "container/internal/container_memory.h"
  "container/internal/counting_allocator.h"
  "container/internal/hash_function_defaults.h"
  "container/internal/hash_policy_traits.h"
  "container/internal/hashtable_debug.h"
  "container/internal/hashtable_debug_hooks.h"
  "container/internal/hashtablez_sampler.cc"
  "container/internal/hashtablez_sampler.h"
  "container/internal/hashtablez_sampler_force_weak_definition.cc"
  "container/internal/inlined_vector.h"
  "container/internal/layout.h"
  "container/internal/node_slot_policy.h"
  "container/internal/raw_hash_map.h"
  "container/internal/raw_hash_set.cc"
  "container/internal/raw_hash_set.h"
  "container/internal/tracked.h"
  "container/node_hash_map.h"
  "container/node_hash_set.h"
  "crc/crc32c.cc"
  "crc/crc32c.h"
  "crc/internal/cpu_detect.cc"
  "crc/internal/cpu_detect.h"
  "crc/internal/crc32c.h"
  "crc/internal/crc32c_inline.h"
  "crc/internal/crc32_x86_arm_combined_simd.h"
  "crc/internal/crc.cc"
  "crc/internal/crc.h"
  "crc/internal/crc_internal.h"
  "crc/internal/crc_x86_arm_combined.cc"
  "crc/internal/crc_memcpy_fallback.cc"
  "crc/internal/crc_memcpy.h"
  "crc/internal/crc_memcpy_x86_64.cc"
  "crc/internal/crc_non_temporal_memcpy.cc"
  "crc/internal/crc_x86_arm_combined.cc"
  "crc/internal/non_temporal_arm_intrinsics.h"
  "crc/internal/non_temporal_memcpy.h"
  "debugging/failure_signal_handler.cc"
  "debugging/failure_signal_handler.h"
  "debugging/leak_check.h"
  "debugging/stacktrace.cc"
  "debugging/stacktrace.h"
  "debugging/symbolize.cc"
  "debugging/symbolize.h"
  "debugging/internal/address_is_readable.cc"
  "debugging/internal/address_is_readable.h"
  "debugging/internal/demangle.cc"
  "debugging/internal/demangle.h"
  "debugging/internal/elf_mem_image.cc"
  "debugging/internal/elf_mem_image.h"
  "debugging/internal/examine_stack.cc"
  "debugging/internal/examine_stack.h"
  "debugging/internal/stack_consumption.cc"
  "debugging/internal/stack_consumption.h"
  "debugging/internal/stacktrace_config.h"
  "debugging/internal/symbolize.h"
  "debugging/internal/vdso_support.cc"
  "debugging/internal/vdso_support.h"
  "functional/any_invocable.h"
  "functional/internal/front_binder.h"
  "functional/bind_front.h"
  "functional/function_ref.h"
  "functional/internal/any_invocable.h"
  "functional/internal/function_ref.h"
  "hash/hash.h"
  "hash/internal/city.h"
  "hash/internal/city.cc"
  "hash/internal/hash.h"
  "hash/internal/hash.cc"
  "hash/internal/spy_hash_state.h"
  "hash/internal/low_level_hash.h"
  "hash/internal/low_level_hash.cc"
  "memory/memory.h"
  "meta/type_traits.h"
  "numeric/bits.h"
  "numeric/int128.cc"
  "numeric/int128.h"
  "numeric/internal/bits.h"
  "numeric/internal/representation.h"
  "profiling/internal/exponential_biased.cc"
  "profiling/internal/exponential_biased.h"
  "profiling/internal/periodic_sampler.cc"
  "profiling/internal/periodic_sampler.h"
  "profiling/internal/sample_recorder.h"
  "random/bernoulli_distribution.h"
  "random/beta_distribution.h"
  "random/bit_gen_ref.h"
  "random/discrete_distribution.cc"
  "random/discrete_distribution.h"
  "random/distributions.h"
  "random/exponential_distribution.h"
  "random/gaussian_distribution.cc"
  "random/gaussian_distribution.h"
  "random/internal/distribution_caller.h"
  "random/internal/fastmath.h"
  "random/internal/fast_uniform_bits.h"
  "random/internal/generate_real.h"
  "random/internal/iostream_state_saver.h"
  "random/internal/mock_helpers.h"
  "random/internal/nonsecure_base.h"
  "random/internal/pcg_engine.h"
  "random/internal/platform.h"
  "random/internal/pool_urbg.cc"
  "random/internal/pool_urbg.h"
  "random/internal/randen.cc"
  "random/internal/randen.h"
  "random/internal/randen_detect.cc"
  "random/internal/randen_detect.h"
  "random/internal/randen_engine.h"
  "random/internal/randen_hwaes.cc"
  "random/internal/randen_hwaes.h"
  "random/internal/randen_round_keys.cc"
  "random/internal/randen_slow.cc"
  "random/internal/randen_slow.h"
  "random/internal/randen_traits.h"
  "random/internal/salted_seed_seq.h"
  "random/internal/seed_material.cc"
  "random/internal/seed_material.h"
  "random/internal/sequence_urbg.h"
  "random/internal/traits.h"
  "random/internal/uniform_helper.h"
  "random/internal/wide_multiply.h"
  "random/log_uniform_int_distribution.h"
  "random/poisson_distribution.h"
  "random/random.h"
  "random/seed_gen_exception.cc"
  "random/seed_gen_exception.h"
  "random/seed_sequences.cc"
  "random/seed_sequences.h"
  "random/uniform_int_distribution.h"
  "random/uniform_real_distribution.h"
  "random/zipf_distribution.h"
  "status/internal/status_internal.h"
  "status/internal/statusor_internal.h"
  "status/status.h"
  "status/status.cc"
  "status/statusor.h"
  "status/statusor.cc"
  "status/status_payload_printer.h"
  "status/status_payload_printer.cc"
  "strings/ascii.cc"
  "strings/ascii.h"
  "strings/charconv.cc"
  "strings/charconv.h"
  "strings/cord.cc"
  "strings/cord.h"
  "strings/cord_analysis.cc"
  "strings/cord_analysis.h"
  "strings/cord_buffer.cc"
  "strings/cord_buffer.h"
  "strings/escaping.cc"
  "strings/escaping.h"
  "strings/internal/charconv_bigint.cc"
  "strings/internal/charconv_bigint.h"
  "strings/internal/charconv_parse.cc"
  "strings/internal/charconv_parse.h"
  "strings/internal/cord_data_edge.h"
  "strings/internal/cord_internal.cc"
  "strings/internal/cord_internal.h"
  "strings/internal/cord_rep_btree.cc"
  "strings/internal/cord_rep_btree.h"
  "strings/internal/cord_rep_btree_navigator.cc"
  "strings/internal/cord_rep_btree_navigator.h"
  "strings/internal/cord_rep_btree_reader.cc"
  "strings/internal/cord_rep_btree_reader.h"
  "strings/internal/cord_rep_crc.cc"
  "strings/internal/cord_rep_crc.h"
  "strings/internal/cord_rep_consume.h"
  "strings/internal/cord_rep_consume.cc"
  "strings/internal/cord_rep_flat.h"
  "strings/internal/cord_rep_ring.cc"
  "strings/internal/cord_rep_ring.h"
  "strings/internal/cord_rep_ring_reader.h"
  "strings/internal/cordz_functions.cc"
  "strings/internal/cordz_functions.h"
  "strings/internal/cordz_handle.cc"
  "strings/internal/cordz_handle.h"
  "strings/internal/cordz_info.cc"
  "strings/internal/cordz_info.h"
  "strings/internal/cordz_sample_token.cc"
  "strings/internal/cordz_sample_token.h"
  "strings/internal/cordz_statistics.h"
  "strings/internal/cordz_update_scope.h"
  "strings/internal/cordz_update_tracker.h"
  "strings/internal/damerau_levenshtein_distance.h"
  "strings/internal/damerau_levenshtein_distance.cc"
  "strings/internal/stl_type_traits.h"
  "strings/internal/string_constant.h"
  "strings/internal/stringify_sink.h"
  "strings/internal/stringify_sink.cc"
  "strings/internal/has_absl_stringify.h"
  "strings/match.cc"
  "strings/match.h"
  "strings/numbers.cc"
  "strings/numbers.h"
  "strings/str_format.h"
  "strings/str_cat.cc"
  "strings/str_cat.h"
  "strings/str_join.h"
  "strings/str_replace.cc"
  "strings/str_replace.h"
  "strings/str_split.cc"
  "strings/str_split.h"
  "strings/string_view.cc"
  "strings/string_view.h"
  "strings/strip.h"
  "strings/substitute.cc"
  "strings/substitute.h"
  "strings/internal/char_map.h"
  "strings/internal/escaping.h"
  "strings/internal/escaping.cc"
  "strings/internal/memutil.cc"
  "strings/internal/memutil.h"
  "strings/internal/ostringstream.cc"
  "strings/internal/ostringstream.h"
  "strings/internal/pow10_helper.cc"
  "strings/internal/pow10_helper.h"
  "strings/internal/resize_uninitialized.h"
  "strings/internal/str_format/arg.cc"
  "strings/internal/str_format/arg.h"
  "strings/internal/str_format/bind.cc"
  "strings/internal/str_format/bind.h"
  "strings/internal/str_format/checker.h"
  "strings/internal/str_format/extension.cc"
  "strings/internal/str_format/extension.h"
  "strings/internal/str_format/float_conversion.cc"
  "strings/internal/str_format/float_conversion.h"
  "strings/internal/str_format/output.cc"
  "strings/internal/str_format/output.h"
  "strings/internal/str_format/parser.cc"
  "strings/internal/str_format/parser.h"
  "strings/internal/str_join_internal.h"
  "strings/internal/str_split_internal.h"
  "strings/internal/utf8.cc"
  "strings/internal/utf8.h"
  "synchronization/barrier.cc"
  "synchronization/barrier.h"
  "synchronization/blocking_counter.cc"
  "synchronization/blocking_counter.h"
  "synchronization/mutex.cc"
  "synchronization/mutex.h"
  "synchronization/notification.cc"
  "synchronization/notification.h"
  "synchronization/internal/create_thread_identity.cc"
  "synchronization/internal/create_thread_identity.h"
  "synchronization/internal/futex.h"
  "synchronization/internal/graphcycles.cc"
  "synchronization/internal/graphcycles.h"
  "synchronization/internal/kernel_timeout.h"
  "synchronization/internal/per_thread_sem.cc"
  "synchronization/internal/per_thread_sem.h"
  "synchronization/internal/thread_pool.h"
  "synchronization/internal/waiter.cc"
  "synchronization/internal/waiter.h"
  "time/civil_time.cc"
  "time/civil_time.h"
  "time/clock.cc"
  "time/clock.h"
  "time/duration.cc"
  "time/format.cc"
  "time/time.cc"
  "time/time.h"
  "time/internal/cctz/include/cctz/civil_time.h"
  "time/internal/cctz/include/cctz/civil_time_detail.h"
  "time/internal/cctz/include/cctz/time_zone.h"
  "time/internal/cctz/include/cctz/zone_info_source.h"
  "time/internal/cctz/src/civil_time_detail.cc"
  "time/internal/cctz/src/time_zone_fixed.cc"
  "time/internal/cctz/src/time_zone_fixed.h"
  "time/internal/cctz/src/time_zone_format.cc"
  "time/internal/cctz/src/time_zone_if.cc"
  "time/internal/cctz/src/time_zone_if.h"
  "time/internal/cctz/src/time_zone_impl.cc"
  "time/internal/cctz/src/time_zone_impl.h"
  "time/internal/cctz/src/time_zone_info.cc"
  "time/internal/cctz/src/time_zone_info.h"
  "time/internal/cctz/src/time_zone_libc.cc"
  "time/internal/cctz/src/time_zone_libc.h"
  "time/internal/cctz/src/time_zone_lookup.cc"
  "time/internal/cctz/src/time_zone_posix.cc"
  "time/internal/cctz/src/time_zone_posix.h"
  "time/internal/cctz/src/tzfile.h"
  "time/internal/cctz/src/zone_info_source.cc"
  "types/any.h"
  "types/bad_any_cast.cc"
  "types/bad_any_cast.h"
  "types/bad_optional_access.cc"
  "types/bad_optional_access.h"
  "types/bad_variant_access.cc"
  "types/bad_variant_access.h"
  "types/compare.h"
  "types/internal/conformance_aliases.h"
  "types/internal/conformance_archetype.h"
  "types/internal/conformance_profile.h"
  "types/internal/parentheses.h"
  "types/internal/transform_args.h"
  "types/internal/variant.h"
  "types/optional.h"
  "types/internal/optional.h"
  "types/span.h"
  "types/internal/span.h"
  "types/variant.h"
  "utility/utility.h"
  "debugging/leak_check.cc"
)

set(ABSL_INTERNAL_DLL_TARGETS
  "algorithm"
  "algorithm_container"
  "any"
  "any_invocable"
  "atomic_hook"
  "bad_any_cast"
  "bad_any_cast_impl"
  "bad_optional_access"
  "bad_variant_access"
  "base"
  "base_internal"
  "bind_front"
  "bits"
  "btree"
  "city"
  "civil_time"
  "compare"
  "compressed_tuple"
  "config"
  "container"
  "container_common"
  "container_memory"
  "cord"
  "core_headers"
  "counting_allocator"
  "crc_cpu_detect"
  "crc_internal"
  "crc32c"
  "debugging"
  "debugging_internal"
  "demangle_internal"
  "dynamic_annotations"
  "endian"
  "examine_stack"
  "exponential_biased"
  "failure_signal_handler"
  "fixed_array"
  "flat_hash_map"
  "flat_hash_set"
  "function_ref"
  "graphcycles_internal"
  "hash"
  "hash_function_defaults"
  "hash_policy_traits"
  "hashtable_debug"
  "hashtable_debug_hooks"
  "hashtablez_sampler"
  "inlined_vector"
  "inlined_vector_internal"
  "int128"
  "kernel_timeout_internal"
  "layout"
  "leak_check"
  "log_severity"
  "malloc_internal"
  "memory"
  "meta"
  "node_hash_map"
  "node_hash_set"
  "node_slot_policy"
  "non_temporal_arm_intrinsics"
  "non_temporal_memcpy"
  "numeric"
  "optional"
  "periodic_sampler"
  "pow10_helper"
  "pretty_function"
  "random_bit_gen_ref"
  "random_distributions"
  "random_internal_distribution_caller"
  "random_internal_distributions"
  "random_internal_explicit_seed_seq"
  "random_internal_fastmath"
  "random_internal_fast_uniform_bits"
  "random_internal_generate_real"
  "random_internal_iostream_state_saver"
  "random_internal_nonsecure_base"
  "random_internal_pcg_engine"
  "random_internal_platform"
  "random_internal_pool_urbg"
  "random_internal_randen"
  "random_internal_randen_engine"
  "random_internal_randen_hwaes"
  "random_internal_randen_hwaes_impl"
  "random_internal_randen_slow"
  "random_internal_salted_seed_seq"
  "random_internal_seed_material"
  "random_internal_sequence_urbg"
  "random_internal_traits"
  "random_internal_uniform_helper"
  "random_internal_wide_multiply"
  "random_random"
  "random_seed_gen_exception"
  "random_seed_sequences"
  "raw_hash_map"
  "raw_hash_set"
  "raw_logging_internal"
  "sample_recorder"
  "scoped_set_env"
  "span"
  "spinlock_wait"
  "spy_hash_state"
  "stack_consumption"
  "stacktrace"
  "status"
  "str_format"
  "str_format_internal"
  "strings"
  "strings_internal"
  "symbolize"
  "synchronization"
  "thread_pool"
  "throw_delegate"
  "time"
  "time_zone"
  "tracked"
  "type_traits"
  "utility"
  "variant"
)

function(_absl_target_compile_features_if_available TARGET TYPE FEATURE)
  if(FEATURE IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(${TARGET} ${TYPE} ${FEATURE})
  else()
    message(WARNING "Feature ${FEATURE} is unknown for the CXX compiler")
  endif()
endfunction()

include(CheckCXXSourceCompiles)

check_cxx_source_compiles(
  [==[
#ifdef _MSC_VER
#  if _MSVC_LANG < 201700L
#    error "The compiler defaults or is configured for C++ < 17"
#  endif
#elif __cplusplus < 201700L
#  error "The compiler defaults or is configured for C++ < 17"
#endif
int main() { return 0; }
]==]
  ABSL_INTERNAL_AT_LEAST_CXX17)

if(ABSL_INTERNAL_AT_LEAST_CXX17)
  set(ABSL_INTERNAL_CXX_STD_FEATURE cxx_std_17)
else()
  set(ABSL_INTERNAL_CXX_STD_FEATURE cxx_std_14)
endif()

function(absl_internal_dll_contains)
  cmake_parse_arguments(ABSL_INTERNAL_DLL
    ""
    "OUTPUT;TARGET"
    ""
    ${ARGN}
  )

  STRING(REGEX REPLACE "^absl::" "" _target ${ABSL_INTERNAL_DLL_TARGET})

  list(FIND
    ABSL_INTERNAL_DLL_TARGETS
    "${_target}"
    _index)

  if (${_index} GREATER -1)
    set(${ABSL_INTERNAL_DLL_OUTPUT} 1 PARENT_SCOPE)
  else()
    set(${ABSL_INTERNAL_DLL_OUTPUT} 0 PARENT_SCOPE)
  endif()
endfunction()

function(absl_internal_dll_targets)
  cmake_parse_arguments(ABSL_INTERNAL_DLL
  ""
  "OUTPUT"
  "DEPS"
  ${ARGN}
  )

  set(_deps "")
  foreach(dep IN LISTS ABSL_INTERNAL_DLL_DEPS)
    absl_internal_dll_contains(TARGET ${dep} OUTPUT _contains)
    if (_contains)
      list(APPEND _deps abseil_dll)
    else()
      list(APPEND _deps ${dep})
    endif()
  endforeach()

  # Because we may have added the DLL multiple times
  list(REMOVE_DUPLICATES _deps)
  set(${ABSL_INTERNAL_DLL_OUTPUT} "${_deps}" PARENT_SCOPE)
endfunction()

function(absl_make_dll)
  add_library(
    abseil_dll
    SHARED
      "${ABSL_INTERNAL_DLL_FILES}"
  )
  target_link_libraries(
    abseil_dll
    PRIVATE
      ${ABSL_DEFAULT_LINKOPTS}
  )
  set_property(TARGET abseil_dll PROPERTY LINKER_LANGUAGE "CXX")
  target_include_directories(
    abseil_dll
    PUBLIC
      "$<BUILD_INTERFACE:${ABSL_COMMON_INCLUDE_DIRS}>"
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  target_compile_options(
    abseil_dll
    PRIVATE
      ${ABSL_DEFAULT_COPTS}
  )

  target_compile_definitions(
    abseil_dll
    PRIVATE
      ABSL_BUILD_DLL
      NOMINMAX
    INTERFACE
      ${ABSL_CC_LIB_DEFINES}
      ABSL_CONSUME_DLL
  )

  if(ABSL_PROPAGATE_CXX_STD)
    # Abseil libraries require C++14 as the current minimum standard. When
    # compiled with C++17 (either because it is the compiler's default or
    # explicitly requested), then Abseil requires C++17.
    _absl_target_compile_features_if_available(${_NAME} PUBLIC ${ABSL_INTERNAL_CXX_STD_FEATURE})
  else()
    # Note: This is legacy (before CMake 3.8) behavior. Setting the
    # target-level CXX_STANDARD property to ABSL_CXX_STANDARD (which is
    # initialized by CMAKE_CXX_STANDARD) should have no real effect, since
    # that is the default value anyway.
    #
    # CXX_STANDARD_REQUIRED does guard against the top-level CMake project
    # not having enabled CMAKE_CXX_STANDARD_REQUIRED (which prevents
    # "decaying" to an older standard if the requested one isn't available).
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${ABSL_CXX_STANDARD})
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  endif()

  install(TARGETS abseil_dll EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endfunction()
