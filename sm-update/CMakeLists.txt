SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "sm-update" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("sm-update")
set_target_properties("sm-update" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("sm-update" "ceed")
