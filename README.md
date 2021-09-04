# NLLK
 
Для компиляции NLLK необходимо указать путь к OpenCV  :
https://github.com/opencv/opencv/releases/tag/4.5.3
https://github.com/opencv/opencv_contrib/releases/tag/4.5.3

После копирования файлов OpenCV в одну папку, для Windows выполнить "setx -m OPENCV_DIR e:\cv453m\build_15\", чтобы студия брала инклюды и либы через $(OPENCV_DIR)
Для Ubuntu подготовлен NLLK\LLK.pro под QT, но не актуализирована с последними изменениями по интеграции YOLO

Собирать OpenCV не обязательно, но если и компилировать, то включить поддержку CUDA.
Если нет поддержки cuda, то до компиляции NLLK использовать флаг YOLO.h::init(tiny flag, cuda flag)

собранные dll с обучалкой DarkNet 
(https://drive.google.com/file/d/1_MXEYU8Tgqv4Q9u-p7fACyuX2FfC9BN1/view?usp=sharing)
 т.к. github плохо работает с файлами больше 50Mb
