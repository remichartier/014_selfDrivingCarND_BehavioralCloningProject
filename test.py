for offset in range(0, 180, 32):        # go from 0 to (num_samples/batch_size)*batch_size (range(start, stop, step))
    # batch_samples = samples[offset:offset+batch_size]
    print(f'From {offset} to {offset+32}')