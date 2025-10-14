#include <stdio.h>

__attribute__((visibility("default")))
void f1() {
  printf("libA: f1()\n");
}

__attribute__((visibility("default")))
void f2() {
  printf("libA: f2()\n");
}

__attribute__((visibility("default")))
void f3() {
  printf("libA: f3()\n");
}
