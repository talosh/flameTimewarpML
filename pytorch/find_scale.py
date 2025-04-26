def generate_sequences():
    results = []

    for a in range(24, 1 - 1, -1):       # First number from 32 down to 2
        for b in range(a, 1 - 1, -1):     # Second number ≤ a
            for c in range(b, 1 - 1, -1): # Third number ≤ b
                for d in range(c, 1 - 1, -1): # Fourth number ≤ c
                    for e in range(d, 1 - 1, -1): # Fifth number ≤ d
                        f = 1  # Sixth number always 1
                        results.append((a, b, c, d, e, f))

    return results

# Example usage:
sequences = generate_sequences()
print(f"Generated {len(sequences)} sequences.")
for seq in sequences[:5]:  # Print first 5 examples
    print(seq)
