SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "sm-update-2" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("sm-update-2")
set_target_properties("sm-update-2" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("sm-update-2" "ceed")
