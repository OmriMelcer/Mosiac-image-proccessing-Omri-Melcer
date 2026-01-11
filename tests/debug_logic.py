
import numpy as np

def test_accumulation_logic():
    print("--- Simulating 5 frames with dx=0.1 ---")
    
    # Setup
    changes = [0.1] * 5
    
    # 1. User's Logic (+=)
    print("\n[User's Logic (+=)]")
    residual_tx = 0.0
    total_pixel_shift = 0
    
    for i, dx in enumerate(changes):
        tx_with_residue = dx + residual_tx
        # Simulate the += line from the code:
        # resedual_tx += tx_with_resedue - round(tx_with_resedue)
        rounded = round(tx_with_residue)
        diff = tx_with_residue - rounded
        residual_tx += diff
        
        total_pixel_shift += rounded
        print(f"Frame {i+1}: dx={dx:.1f}, Current Total={tx_with_residue:.2f}, Round={rounded}, New Resid={residual_tx:.2f}")

    print(f"Total Pixels Shifted: {total_pixel_shift} (Expected for 0.5 accumulated: 0 or 1 depending on rounding)")

    # 2. Standard Logic (=)
    print("\n[Standard Logic (=)]")
    residual_tx = 0.0
    total_pixel_shift = 0
    
    for i, dx in enumerate(changes):
        tx_with_residue = dx + residual_tx
        
        rounded = round(tx_with_residue)
        # Using assignment =
        residual_tx = tx_with_residue - rounded
        
        total_pixel_shift += rounded
        print(f"Frame {i+1}: dx={dx:.1f}, Current Total={tx_with_residue:.2f}, Round={rounded}, New Resid={residual_tx:.2f}")

    print(f"Total Pixels Shifted: {total_pixel_shift}")

if __name__ == "__main__":
    test_accumulation_logic()
