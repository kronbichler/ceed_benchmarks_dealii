SET(TARGET_SRC multigrid_cg.cc)

ADD_EXECUTABLE( "bps5_evec" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("bps5_evec")
set_target_properties("bps5_evec" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("bps5_evec" "ceed")
