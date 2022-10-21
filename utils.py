import torch


def merge_input_and_gen_ids(input_ids, generated_ids, pad_id=0, eos_id=1, idx_a=32000, idx_b=32099):
    new_input_ids = []
    for k in range(len(input_ids)):
        inp_ids = input_ids[k]
        inp_len = (inp_ids != pad_id).sum().item()
        gen_ids = generated_ids[k]
        gen_len = (gen_ids != pad_id).sum().item()
        z_x = ~((inp_ids >= idx_a) & (inp_ids <= idx_b))
        z_x = z_x & (inp_ids != pad_id) & (inp_ids != eos_id)
        z_x = z_x.long().tolist()
        z_y = ((gen_ids >= idx_a) & (gen_ids <= idx_b))
        z_y = z_y & (gen_ids != pad_id) & (gen_ids != eos_id)
        z_y = z_y.long().tolist()
        i, j = 0, 0
        new_inp = []
        while j < gen_len:
            if z_y[j] == 1:
                while z_x[i] == 1 and i < inp_len:
                    new_inp.append(inp_ids[i].item())
                    i += 1
                j += 1
                i += 1
                if i >= inp_len:
                    break
            else:
                new_inp.append(gen_ids[j].item())
                j += 1
        if i < inp_len:
            new_inp.extend(inp_ids[i:inp_len])
        if new_inp[-1] != 1 and new_inp[-1] != 0:
            new_inp.append(1)
        new_input_ids.append(torch.as_tensor(new_inp))
    x_new = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=pad_id)
    x_new = x_new.to(input_ids.device)
    return x_new
