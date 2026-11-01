"""MO433 project: is distilling a frozen self-supervised ViT into a small convnet (row C)
better than training the same convnet directly with VICReg (row B)?

Every step of the pipeline is a module runnable as ``python -m mo433.<step>``; the
Makefile wires them together. Data and outputs live in ``$MO433_DATA`` (see ``mo433.io``).
"""
