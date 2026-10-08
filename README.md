
### Setup:

- lr rate(start) = 2e-3 (start, using lr schelduer)
- picture size = 96*128
- Epoch per batch = 10/67
- Data split = 70% train / 15% validation / 15% test
- Frameworks - PyTorch, SnnTorch, Tonic
- Seed = 42 (Torch, NumPy, cuda)
- Dataset - NCaltech101 (from Tonic)
- Input - 20 time bins x 2 polarities x 96 x 128 (H x W)
- 
### Architektura: 

Conv2d(2 -> 16, 3x3, pad 1, no bias) -> MaxPool 2x2   -> 16 x 48 x 64
  -> Leaky LIF neurons (leaky1, beta 0.9)                   -> spikes
  -> Conv2d(16 -> 32, 3x3, pad 1)         -> MaxPool 2x2   -> 32 x 24 x 32
  -> Leaky LIF neurons (leaky2, beta 0.9)                   -> spikes
  -> Flatten (24576) -> Dropout(0.5) -> Linear(24576 -> 101)
  -> Leaky LIF output layer (leaky_out, beta 0.9)
  
### Augmentacja:

RandomFlipPolarity
TimeJitter
