# NLLK
 
для компиляции иметь https://github.com/opencv/opencv/releases/tag/4.5.3
+
https://github.com/opencv/opencv_contrib/releases/tag/4.5.3
если компилировать то включить поддержку CUDA
если нет поддержки cuda то использовать флаг YOLO.h::init(tiny flag, cuda flag)

после кописрования файлов OpenCV в одну папку, для Windows выполнить "setx -m OPENCV_DIR e:\cv453m\build_15\", чтобы студия брала инклюды и либы через $(OPENCV_DIR)

собранные dll с обучалкой DarkNet выгружу отдельно архивом т.к. github плохо работет с файлами больше 50Mb
