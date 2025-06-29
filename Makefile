# Use mingw32-make as the command, not make :/

# Compiler
CC = gcc

# Compiler flags (note the backslashes for continuation)
CFLAGS = -Wall -Wextra -Werror -Wfloat-equal -Wundef -Wshadow -Wpointer-arith -Wcast-align \
-Wstrict-prototypes -Wwrite-strings -Waggregate-return \
-Wcast-qual -Wswitch-default -Wswitch-enum -Wconversion -Wunreachable-code \
-Isrc -Ofast

# Source files
SRC = main.c src/ann.c

# Output executable
TARGET = main.exe

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean target
.PHONY: clean
clean:
	del /Q $(TARGET)