#include "ForwardTestRun.h"
using namespace std;
ForwardTestRun::ForwardTestRun(){}
ForwardTestRun::ForwardTestRun(const JsonValue& j): ForwardBackwardRun(j){
}
void ForwardTestRun::operator()(Net004& net, int cur) {
	printf("[%d]run: test\n",cur);
}
