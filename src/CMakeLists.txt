add_library(radial SHARED
  recovery_common.cc
  solution_recovery.cc
  )

deal_ii_setup_target(radial)
