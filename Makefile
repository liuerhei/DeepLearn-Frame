# compiler
#
CXX = nvcc
CXXFLAGS = -arch=sm_35 -std=c++11 -O2 -g -Xcompiler -fmax-errors=1 -Xcompiler -g3
LD = nvcc
LDFLAGS = -arch=sm_35 -lcuda -lcudnn -lcublas

# project
#
#run: main.o wheel.o session.o loss.o					
run: test_fc.o wheel.o session.o loss.o					\
	tensor/itensor.o operator/ioperator.o  				\
	operator/conv2d.o operator/pooling2d.o operator/activation2d.o operator/softmax.o 				\
	operator/fc2d.o	operator/batchnormalization2d.o												\
	tensor/tensor4d.o tensor/filter4d.o                                                                             \
	readubyte.o
	$(LD) $(LDFLAGS) -o $@ $^

operator/conv2d.o: operator/conv2d.cu 
	 $(CXX) $(CXXFLAGS)   -c -o $@ $^
operator/fc2d.o: operator/fc2d.cu 
	 $(CXX) $(CXXFLAGS)   -c -o $@ $^
operator/fc2d_test.o: operator/fc2d_test.cu 
	 $(CXX) $(CXXFLAGS)   -c -o $@ $^
loss.o: loss.cu
	 $(CXX) $(CXXFLAGS)   -c -o $@ $^
main.o: main.cu
	 $(CXX) $(CXXFLAGS)   -c -o $@ $^
#
# phony target
#
.PHONY: all
all: run

.PHONY: clean
clean:
	rm -rf *.o && rm -rf operator/*.o && rm -rf tensor/*.o && rm -rf run

