"""
Generate shared test instances for fair DRL vs MOEA/D comparison.
Produces: paretofront/shared_test_50.pkl and paretofront/shared_test_100.pkl
Each file contains 1000 instances in the same format as AGHDataset.
"""
import os
import pickle
import numpy as np
import torch

def generate_shared_test(size, num_samples=1000, seed=7654321):
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_hour = np.arange(10, 20)
    n_min = 60
    n_gate = 100
    prob = np.load('problems/agh/arrival_prob.npy')

    loc = 1 + np.random.choice(n_gate, size=(num_samples, size))
    arrival = (60 * np.random.choice(n_hour, size=(num_samples, size), p=prob)
               + np.random.randint(0, n_min, size=(num_samples, size)))
    stay = torch.tensor([120, 120, 120]).repeat(num_samples, 1)
    type_ = torch.tensor(np.random.randint(0, 3, size=(num_samples, size)), dtype=torch.long)
    departure = torch.tensor(arrival) + torch.gather(stay, 1, type_)

    service_to_select = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    prob_need = torch.tensor([0.116, 0.116, 0.116, 0.116, 0.116, 0.116, 0.1, 0.1, 0.1])
    need = torch.stack([
        service_to_select[torch.multinomial(prob_need, size, replacement=True)]
        for _ in range(num_samples)
    ])

    data = list(zip(loc.tolist(), arrival.tolist(), departure.tolist(), type_.tolist(), need.tolist()))
    return data


def main():
    os.makedirs('paretofront', exist_ok=True)

    for size in [50, 100]:
        data = generate_shared_test(size)
        out_path = f'paretofront/shared_test_{size}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[OK] Generated {out_path}: {len(data)} instances, n={size}")


if __name__ == '__main__':
    main()
