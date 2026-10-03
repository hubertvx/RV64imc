    .section .text.init
    .globl _start
_start:
    la      sp, _stack_top
    la      t0, _bss_start
    la      t1, _bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:  call    main
    call    runtime_exit
3:  j       3b
