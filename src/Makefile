APP_NAME = ballflight
OBJS = ballflight.o

CXX = mpic++
CXXFLAGS = -Wall -Wextra -O3 -std=c++20 -I.

default: $(APP_NAME)

$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	/bin/rm -rf *~ *.o $(APP_NAME) *.class

FILES= *.h \
	*.cpp 
