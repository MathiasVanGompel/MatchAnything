#!/usr/bin/env python3
import onnx, argparse
from onnx import numpy_helper, helper, TensorProto

def node_by_output(graph):
    mp = {}
    for n in graph.node:
        for o in n.output:
            mp[o] = n
    return mp

def get_const_tensor_from_node(n):
    # Return numpy array if node is Constant with a "value" attribute, else None
    if n.op_type != "Constant":
        return None
    for a in n.attribute:
        if a.name == "value" and a.t is not None:
            return numpy_helper.to_array(a.t)
    return None

def set_const_tensor_on_node(n, np_value, dtype=None):
    # Replace Constant's value with np_value (broadcastable or scalar ok)
    if n.op_type != "Constant":
        return False
    new_t = numpy_helper.from_array(np_value.astype(dtype) if dtype is not None else np_value)
    # rewrite attribute
    for i,a in enumerate(n.attribute):
        if a.name == "value":
            n.attribute[i].t.CopyFrom(new_t)
            return True
    # if no value attribute, add one
    n.attribute.extend([helper.make_attribute("value", new_t)])
    return True

def follow_simple_chain(prod_map, name, max_hops=3):
    """Follow through Cast/Reshape/Squeeze/Unsqueeze to find an upstream Constant, if any."""
    cur = name
    hops = 0
    path = []
    while cur in prod_map and hops < max_hops:
        n = prod_map[cur]
        path.append(n)
        if n.op_type in ("Cast","Reshape","Squeeze","Unsqueeze","Identity"):
            # follow first data input
            if len(n.input) >= 1 and n.input[0]:
                cur = n.input[0]
                hops += 1
                continue
        break
    return path  # includes the last non-followed node if any

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_in", required=True)
    ap.add_argument("--onnx_out", required=True)
    ap.add_argument("--thr", type=float, required=True, help="new threshold for Greater()")
    args = ap.parse_args()

    model = onnx.load(args.onnx_in)
    g = model.graph
    prod = node_by_output(g)

    touched = 0
    for n in g.node:
        if n.op_type != "Greater":
            continue

        # Try each input of Greater: look for Constant directly or via simple chain
        for inp in list(n.input):
            # direct Constant?
            if inp in prod and prod[inp].op_type == "Constant":
                cnode = prod[inp]
                carr = get_const_tensor_from_node(cnode)
                if carr is not None:
                    import numpy as np
                    new_val = np.array(args.thr, dtype=carr.dtype)
                    ok = set_const_tensor_on_node(cnode, new_val)
                    if ok:
                        touched += 1
                        print(f"[OK] Rewrote Constant feeding Greater '{n.name or ''}' to {args.thr}")
                        break
            # follow Cast/Reshape/... -> Constant
            chain = follow_simple_chain(prod, inp, max_hops=4)
            if chain:
                last = chain[-1]
                if last.op_type == "Constant":
                    carr = get_const_tensor_from_node(last)
                    if carr is not None:
                        import numpy as np
                        new_val = np.array(args.thr, dtype=carr.dtype)
                        ok = set_const_tensor_on_node(last, new_val)
                        if ok:
                            touched += 1
                            print(f"[OK] Rewrote Constant via chain -> Greater '{n.name or ''}' to {args.thr}")
                            break

    if touched == 0:
        print("[WARN] Did not find a Constant/initializer feeding any Greater node.")
    else:
        print(f"[INFO] Updated {touched} Constant(s) feeding Greater.")

    onnx.checker.check_model(model)
    onnx.save(model, args.onnx_out)
    print(f"[OK] Wrote patched ONNX -> {args.onnx_out}")

if __name__ == "__main__":
    main()
