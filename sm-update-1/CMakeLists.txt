SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "sm-update-1" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("sm-update-1")
set_target_properties("sm-update-1" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("sm-update-1" "ceed")
