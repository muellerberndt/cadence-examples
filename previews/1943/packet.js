const types = {f4:Float32Array,f8:Float64Array,u4:Uint32Array,i4:Int32Array,u1:Uint8Array};
export function decode(buffer) {
  const v=new DataView(buffer);
  if(buffer.byteLength<8 || v.getUint32(0,true)!==0x31524443) throw Error('Invalid recording packet');
  const n=v.getUint32(4,true);
  if(n>buffer.byteLength-8) throw Error('Truncated recording header');
  const meta=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,8,n)));
  const start=8+n+(8-n%8)%8, arrays={};
  for(const [name,a] of Object.entries(meta.arrays)) {
    const T=types[a.dtype], count=a.shape.reduce((x,y)=>x*y,1);
    if(!T || !a.shape.every(x=>Number.isSafeInteger(x)&&x>=0) || !Number.isSafeInteger(count) || !Number.isSafeInteger(a.offset) || a.offset<0 || a.offset%8 || a.bytes!==count*T.BYTES_PER_ELEMENT || start+a.offset+a.bytes>buffer.byteLength) throw Error('Invalid recording array');
    if(a.encoding){
      if(a.encoding!=='xor-rows'||a.shape.length!==2||!['f4','f8'].includes(a.dtype))throw Error('Unsupported recording encoding');
      const words=new Uint32Array(buffer,start+a.offset,a.bytes/4),stride=a.shape[1]*T.BYTES_PER_ELEMENT/4;
      for(let i=stride;i<words.length;i++)words[i]^=words[i-stride];
    }
    if(a.reference){
      const ref=arrays[a.reference];
      if(!ref||ref.constructor!==T||ref.length!==count)throw Error('Invalid recording reference');
      const words=new Uint8Array(buffer,start+a.offset,a.bytes),base=new Uint8Array(ref.buffer,ref.byteOffset,ref.byteLength);
      for(let i=0;i<words.length;i++)words[i]^=base[i];
    }
    arrays[name]=new T(buffer,start+a.offset,count);
  }
  return {meta,arrays};
}
export function restoreBase(data,base) {
  for(const [name,value] of Object.entries(data.arrays)) {
    const reference=base.arrays[name];
    if(!reference||reference.constructor!==value.constructor||reference.length!==value.length||JSON.stringify(data.meta.arrays[name].shape)!==JSON.stringify(base.meta.arrays[name].shape))throw Error('Invalid base snapshot');
    const count=Math.floor(value.byteLength/4),words=new Uint32Array(value.buffer,value.byteOffset,count),other=new Uint32Array(reference.buffer,reference.byteOffset,count);
    for(let i=0;i<count;i++)words[i]^=other[i];
    const tail=new Uint8Array(value.buffer,value.byteOffset+count*4,value.byteLength%4),refTail=new Uint8Array(reference.buffer,reference.byteOffset+count*4,reference.byteLength%4);
    for(let i=0;i<tail.length;i++)tail[i]^=refTail[i];
  }
  return data;
}
async function load(path) {
  const r=await fetch(new URL('./recording/'+path,import.meta.url));
  if(!r.ok) throw Error(`Cannot load ${path}: ${r.status}`);
  let buffer=await r.arrayBuffer();
  const magic=new Uint8Array(buffer,0,Math.min(2,buffer.byteLength));
  // Pages serves .gz as a file; the local replay server supplies Content-Encoding.
  // Test the bytes to support both without decompressing an HTTP-decoded response twice.
  if(magic[0]===0x1f&&magic[1]===0x8b){
    if(typeof DecompressionStream==='undefined')throw Error('This browser cannot decompress the brain recording. Use a current browser or watch the gameplay video below.');
    buffer=await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  }
  return decode(buffer);
}
const bases=new Map();
export async function packet(path) {
  const data=await load(path),base=data.meta.base;
  if(!base)return data;
  if(!/^(memory|weights)\/\d+\.bin\.gz$/.test(base)||base.split('/')[0]!==path.split('/')[0]||base===path)throw Error('Invalid base snapshot path');
  if(!bases.has(base)){
    const pending=load(base).then(value=>{if(value.meta.base)throw Error('Chained base snapshots are unsupported');return value;}).catch(error=>{bases.delete(base);throw error;});
    bases.set(base,pending);
  }
  return restoreBase(data,await bases.get(base));
}
