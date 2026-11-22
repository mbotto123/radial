add_library(radial SHARED
  recovery_common.cc
  recovery_discrete.cc
  recovery_integral.cc
  recovery.cc
  )

deal_ii_setup_target(radial)
