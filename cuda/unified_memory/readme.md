The benefit for unified memory, in terms of performance, is that it avoids deep copies. <br>
=

Reference:
https://github.com/parallel-forall/code-samples/tree/master/posts/unified-memory


According to CUDA Programming Guide, here is the description of using "cudaMallocManaged()".<br>
--
<i>
On a multi-GPU system with peer-to-peer support, where multiple GPUs support 
managed memory, <b>the physical storage is created on the GPU</b> which is active at 
the time cudaMallocManaged is called. 
All other GPUs will reference the data at reduced bandwidth via peer mappings over the PCIe bus. 
The Unified Memory management system <b>does not migrate memory between GPUs</b>.
</i>

