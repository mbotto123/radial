add_library(radial SHARED
  recovery_common.cc
  recovery_discrete.cc
  recovery_integral.cc
  solution_recovery.cc
  )

deal_ii_setup_target(radial)
