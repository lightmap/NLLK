TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
QT += core
CONFIG(debug, debug|release) {
    #LIBS += -lsocket
    #unix:!macx: LIBS += -lopencv_world
}
LIBS += -L$$PWD/../opencv-4.1.0/Build/lib/ -lopencv_world
MOC_DIR += ./GeneratedFiles/$(ConfigurationName)
OBJECTS_DIR += release
UI_DIR += ./GeneratedFiles
RCC_DIR += ./GeneratedFiles
SOURCES += ./LLK.cpp

INCLUDEPATH += /usr/local/include/opencv4/

HEADERS += \
    camera.h
