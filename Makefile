#Author         :yuqj
#Email          :BrixLab
#Last_update    :2020.12.02

CC              := g++
SRC_INCLUDE     := -I ./include \
                   -I /usr/local/include
LIB_DIR         := lib
SRC_DIR         := ./source/
OBJ_DIR         := ./obj
TARGE_DIR       := ./bin

SOURCES          := $(wildcard $(SRC_DIR)*.c)
SOURCES          += $(wildcard $(SRC_DIR)*.cpp)

ALL_OBJS         := $(patsubst $(SRC_DIR), $(OBJ_DIR)/%.o, $(SOURCES))
#ALL_OBJS         += $(patsubst $(SRC_DIR)%.c, $(OBJ_DIR)/%.o, $(SOURCES))

$(warning $(ALL_OBJS))
DEPS        := ${SOURCES:.c=.d} #.d文件是包含文件的依赖关系不必手动书写若干目标文件的依赖关系，由编译器自动生成，不管是源文件还是头文件有更新，目标文件都会重新编译
DEPS        += ${SOURCES:.cpp=.d}
TARGET          := $(TARGE_DIR)/demo_inference

CFLAGS = -g -Wall -std=c++11
LFLAGS = -L ${LIB_DIR} -ldnnl

.PHONY:all clean
all: ${TARGET}

${TARGET}:${ALL_OBJS}
	if [ ! -d $(TARGE_DIR) ]; then mkdir -p $(TARGE_DIR); fi;
	${CC} ${CFLAGS} ${SRC_INCLUDE} -o ${TARGET} ${ALL_OBJS} ${LFLAGS}

%.d: %.c
	@set -e; rm -f $@; \
	$(CC) -M $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

%.d: %.cpp
	@set -e; rm -f $@; \
	$(CC) -M $(CFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(OBJ_DIR)/%.o:$(SOURCES)
	if [ ! -d $(OBJ_DIR) ]; then mkdir -p $(OBJ_DIR); fi;
	${CC} ${CFLAGS} ${SRC_INCLUDE} -c $< -o $@


clean:
	rm -rf *.o *.d ${TARGET} *~ ${OBJ_DIR} *~