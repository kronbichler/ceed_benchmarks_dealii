SET(TARGET_SRC multigrid_cg.cc)

ADD_EXECUTABLE( "bps5" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("bps5")
set_target_properties("bps5" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("bps5" "ceed")
