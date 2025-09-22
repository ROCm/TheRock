#include <stdio.h>

__attribute__((visibility("default")))
void f2() {
  printf("libB: f2()\n");
}

__attribute__((visibility("default")))
void f3() {
  printf("libB: f3()\n");
}

__attribute__((visibility("default")))
void f4() {
  printf("libB: f4()\n");
}
