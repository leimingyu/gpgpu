The benefit for unified memory, in terms of performance, is that it avoids deep copies.

Reference:
https://github.com/parallel-forall/code-samples/tree/master/posts/unified-memory

According to CUDA Programming Guide
<i>
On a multi-GPU system with peer-to-peer support, where multiple GPUs support 
managed memory, <b>the physical storage is created on the GPU</b> which is active at 
the time cudaMallocManaged is called. 
All other GPUs will reference the data at reduced bandwidth via peer mappings over the PCIe bus. 
The Unified Memory management system <b>does not migrate memory between GPUs</b>.

On a multi-GPU system where multiple GPUs support managed memory, but not all 
pairs of such GPUs have peer-to-peer support between them, the physical storage 
is created in 'zero-copy' or system memory. 
All GPUs will reference the data at reduced bandwidth over the PCIe bus. 
<b>
In these circumstances, use of the environment variable, CUDA_VISIBLE_DEVICES, 
is recommended to restrict CUDA to only use those GPUs that have peer-to-peer support. 
</b>
Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a non-zero 
value to force the driver to always use device memory for physical storage. 
When this environment variable is set to a non-zero value, all devices used in 
that process that support managed memory have to be peer-to-peer compatible with each other.

</i>

