import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import jit
import colorsys

@jit(nopython=True)
def newton_iteration(z0, max_iter=600, tolerance=1e-6):
    """
    Newton's method for f(z) = z^2 * sin(z) - 1
    f'(z) = 2z*sin(z) + z^2*cos(z)
    """
    z = z0
    
    for n in range(max_iter):
        sin_z = np.sin(z)
        cos_z = np.cos(z)
        
        f_z = z * z * sin_z - 1.0
        f_prime_z = 2.0 * z * sin_z + z * z * cos_z
        
        if abs(f_prime_z) < 1e-10:
            return n, z
        
        z_new = z - f_z / f_prime_z
        
        if abs(z_new - z) < tolerance:
            return n, z_new
        
        z = z_new
    
    return max_iter, z

@jit(nopython=True)
def compute_fractal_data(width, height, xmin, xmax, ymin, ymax, max_iter, tolerance):
    """
    Compute fractal with detailed convergence information
    """
    iterations = np.zeros((height, width), dtype=np.float64)
    roots_real = np.zeros((height, width), dtype=np.float64)
    roots_imag = np.zeros((height, width), dtype=np.float64)
    
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    
    for i in range(height):
        y = ymin + i * dy
        for j in range(width):
            x = xmin + j * dx
            z0 = complex(x, y)
            
            iters, root = newton_iteration(z0, max_iter, tolerance)
            
            # Store iteration count with smooth coloring
            iterations[i, j] = iters
            roots_real[i, j] = root.real
            roots_imag[i, j] = root.imag
    
    return iterations, roots_real, roots_imag

def generate_newton_fractal(width=1600, height=1200, xmin=-2.5, xmax=2.5, 
                           ymin=-1.875, ymax=1.875, max_iter=600, tolerance=1e-6):
    """
    Generate high-resolution Newton fractal
    """
    print(f"Generating {width}x{height} Newton fractal...")
    print("Computing iterations... (this will take a few minutes)")
    
    iterations, roots_real, roots_imag = compute_fractal_data(
        width, height, xmin, xmax, ymin, ymax, max_iter, tolerance
    )
    
    print("Fractal computation complete!")
    return iterations, roots_real, roots_imag

def classify_roots(roots_real, roots_imag, tolerance=0.05):
    """
    Classify convergence basins
    """
    print("\nClassifying convergence basins...")
    
    roots_complex = roots_real + 1j * roots_imag
    basin_map = np.zeros(roots_complex.shape, dtype=np.int32)
    unique_roots = []
    
    flat_roots = roots_complex.flatten()
    unique_candidates = []
    
    # Sample subset for unique root detection
    sample_indices = np.random.choice(len(flat_roots), min(10000, len(flat_roots)), replace=False)
    
    for idx in sample_indices:
        root = flat_roots[idx]
        found = False
        for ur in unique_candidates:
            if abs(root - ur) < tolerance:
                found = True
                break
        if not found:
            unique_candidates.append(root)
    
    unique_roots = unique_candidates
    print(f"Found {len(unique_roots)} distinct roots")
    
    # Classify all pixels
    for i in range(roots_complex.shape[0]):
        for j in range(roots_complex.shape[1]):
            root = roots_complex[i, j]
            min_dist = float('inf')
            min_idx = 0
            
            for idx, ur in enumerate(unique_roots):
                dist = abs(root - ur)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            
            basin_map[i, j] = min_idx
    
    return basin_map, unique_roots

def create_enhanced_visualization(iterations, basin_map, num_roots):
    """
    Create the glowing web visualization
    """
    height, width = iterations.shape
    
    # Normalize iterations for better visibility
    iter_normalized = iterations / np.max(iterations)
    
    # Create edge detection for basin boundaries (this creates the web structure)
    from scipy import ndimage
    
    # Detect edges in basin map
    edges_x = ndimage.sobel(basin_map.astype(float), axis=1)
    edges_y = ndimage.sobel(basin_map.astype(float), axis=0)
    edges = np.hypot(edges_x, edges_y)
    edges = edges / np.max(edges) if np.max(edges) > 0 else edges
    
    # Combine iteration data with edge information
    # Areas near boundaries (edges) should be brighter
    brightness = np.zeros_like(iter_normalized)
    
    # Method 1: Edge-based brightness (creates the web)
    brightness += edges * 3.0
    
    # Method 2: Iteration-based coloring (slow convergence = brighter)
    slow_convergence = 1.0 - np.exp(-iter_normalized * 5)
    brightness += slow_convergence * 0.5
    
    # Method 3: Enhance areas where iteration count changes rapidly
    iter_gradient_x = ndimage.sobel(iterations, axis=1)
    iter_gradient_y = ndimage.sobel(iterations, axis=0)
    iter_gradient = np.hypot(iter_gradient_x, iter_gradient_y)
    iter_gradient = iter_gradient / np.max(iter_gradient) if np.max(iter_gradient) > 0 else iter_gradient
    brightness += iter_gradient * 2.0
    
    # Normalize final brightness
    brightness = np.clip(brightness, 0, 1)
    
    # Apply power law to enhance bright features
    brightness = np.power(brightness, 0.7)
    
    return brightness

def plot_fractal_enhanced(iterations, roots_real, roots_imag, width, height):
    """
    Create high-quality visualization with web structure
    """
    print("\nClassifying basins...")
    basin_map, unique_roots = classify_roots(roots_real, roots_imag, tolerance=0.1)
    
    print("Creating enhanced visualization...")
    brightness = create_enhanced_visualization(iterations, basin_map, len(unique_roots))
    
    # Create red-glow colormap with more dynamic range
    colors = [
        (0.0, 0.0, 0.0),      # Black
        (0.05, 0.0, 0.0),     # Very dark red
        (0.1, 0.0, 0.0),
        (0.15, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.3, 0.0, 0.0),
        (0.4, 0.0, 0.0),
        (0.5, 0.0, 0.0),      # Dark red
        (0.6, 0.0, 0.0),
        (0.7, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (0.9, 0.0, 0.0),
        (1.0, 0.0, 0.0),      # Pure red
        (1.0, 0.2, 0.0),
        (1.0, 0.4, 0.0),
        (1.0, 0.6, 0.0),      # Orange
        (1.0, 0.8, 0.0),
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 1.0, 0.5),      # Bright yellow
        (1.0, 1.0, 1.0),      # White (for brightest points)
    ]
    
    cmap = LinearSegmentedColormap.from_list('red_web', colors, N=1024)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='black', dpi=150)
    ax.set_facecolor('black')
    
    # Plot with high-quality interpolation
    im = ax.imshow(brightness, cmap=cmap, interpolation='bicubic',
                   extent=[-2.5, 2.5, -1.875, 1.875], origin='lower',
                   aspect='auto', vmin=0, vmax=1)
    
    # Remove axes
    ax.axis('off')
    
    # Add formula
    ax.text(0.5, 0.02, r'$p(z) = z^{2} \cdot \sin(z) - 1$, $a=1$', 
            transform=ax.transAxes, fontsize=22, color='white',
            ha='center', family='serif')
    
    plt.tight_layout(pad=0)
    
    return fig

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("Newton Fractal Generator - Enhanced Web Visualization")
    print("f(z) = z^2 * sin(z) - 1 = 0")
    print("="*70)
    
    # High-resolution settings
    WIDTH = 1600
    HEIGHT = 1200
    MAX_ITER = 600
    TOLERANCE = 1e-6
    
    # Generate fractal data
    iterations, roots_real, roots_imag = generate_newton_fractal(
        width=WIDTH, 
        height=HEIGHT, 
        xmin=-2.5, 
        xmax=2.5,
        ymin=-1.875, 
        ymax=1.875,
        max_iter=MAX_ITER,
        tolerance=TOLERANCE
    )
    
    # Create enhanced visualization
    fig = plot_fractal_enhanced(iterations, roots_real, roots_imag, WIDTH, HEIGHT)
    
    # Save
    output_file = 'newton_fractal_web_enhanced.png'
    print(f"\nSaving to '{output_file}'...")
    fig.savefig(output_file, dpi=200, facecolor='black', 
                bbox_inches='tight', pad_inches=0.05)
    
    print(f"Saved successfully!")
    print("\nDisplaying...")
    plt.show()
    
    print("\n✨ Done!")
