# compiler
#
CXX = nvcc
CXXFLAGS = -arch=sm_35 -std=c++11 -O2 -g -Xcompiler -fmax-errors=1 -Xcompiler -g3
LD = nvcc
LDFLAGS = -arch=sm_35 -lcuda -lcudnn -lcublas

# project
#
run: main.o wheel.o tensor/itensor.o operator/ioperator.o session.o \
	operator/conv2d.o 					    \
	tensor/tensor4d.o tensor/filter4d.o
	$(LD) $(LDFLAGS) -o $@ $^

# .cu file
#
# operator/upsampling3d.o: operator/upsampling3d.cu
#	 $(CXX) $(CXXFLAGS) -c -o $@ $^
#
# phony target
#
.PHONY: all
all: run

.PHONY: clean
clean:
	rm -rf *.o && rm -rf operator/*.o && rm -rf tensor/*.o && rm -rf run
