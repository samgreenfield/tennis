#!/usr/bin/env python3
"""
Conceptual visualization of the perspective correction problem and solution.
This creates a simple diagram showing why perspective correction is needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def create_perspective_diagram():
    """Create a diagram showing the perspective correction concept"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # === LEFT PLOT: 3D PERSPECTIVE (PROBLEM) ===
    ax1.set_title('PROBLEM: 3D Ball Path → 2D Projection\n(What the camera sees)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    
    # Draw court
    court = Rectangle((1, 1), 8, 3, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3)
    ax1.add_patch(court)
    ax1.text(5, 2.5, 'Tennis Court\n(Top View)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw 3D ball trajectory (parabolic)
    x_3d = np.linspace(1.5, 8.5, 100)
    y_3d_high = 5 + 1.5 * np.sin(np.pi * (x_3d - 1.5) / 7)  # High parabolic path
    
    ax1.plot(x_3d, y_3d_high, 'b-', linewidth=3, label='3D Ball Path (parabolic)')
    
    # Draw projected path (curved on court)
    x_proj = np.linspace(1.5, 8.5, 50)
    y_proj = 1.5 + 0.8 * np.sin(np.pi * (x_proj - 1.5) / 7)  # Projected curve
    
    ax1.plot(x_proj, y_proj, 'r--', linewidth=3, label='Projected Path (curved)')
    
    # Mark bounce points
    bounce_points_x = [2.5, 5, 7.5]
    bounce_points_y = [1.5, 1.5, 1.5]
    ax1.scatter(bounce_points_x, bounce_points_y, c='orange', s=100, zorder=5, label='Actual Bounces')
    
    # Draw projection lines
    for i, x in enumerate([2.5, 5, 7.5]):
        y_high = 5 + 1.5 * np.sin(np.pi * (x - 1.5) / 7)
        ax1.plot([x, x], [y_high, 1.5], 'k:', alpha=0.5, linewidth=1)
        ax1.annotate('', xy=(x, 1.5), xytext=(x, y_high), 
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    ax1.text(5, 6.5, '3D Ball Flight', ha='center', fontsize=12, color='blue', fontweight='bold')
    ax1.text(5, 0.5, 'Camera sees curved projection', ha='center', fontsize=10, color='red')
    
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Court Width')
    ax1.set_ylabel('Height / Court Length')
    ax1.grid(True, alpha=0.3)
    
    # === RIGHT PLOT: CORRECTED VIEW (SOLUTION) ===
    ax2.set_title('SOLUTION: Perspective-Corrected View\n(True top-down view)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    
    # Draw court
    court2 = Rectangle((1, 1), 8, 6, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3)
    ax2.add_patch(court2)
    ax2.text(5, 4, 'Tennis Court\n(True Top View)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw straight-line trajectory between bounces
    bounce_x = [2, 4, 6, 8]
    bounce_y = [2, 6, 3, 5]
    
    # Draw straight line segments
    for i in range(len(bounce_x) - 1):
        ax2.plot([bounce_x[i], bounce_x[i+1]], [bounce_y[i], bounce_y[i+1]], 
                'g-', linewidth=4, alpha=0.8)
    
    # Mark bounce points
    ax2.scatter(bounce_x, bounce_y, c='orange', s=150, zorder=5, edgecolor='black', linewidth=2)
    
    # Add annotations
    for i, (x, y) in enumerate(zip(bounce_x, bounce_y)):
        ax2.annotate(f'Bounce {i+1}', (x, y), xytext=(10, 10), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Add arrows showing straight trajectories
    ax2.annotate('Straight trajectory\nbetween bounces', 
                xy=(3, 4), xytext=(1.5, 7),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=11, fontweight='bold', color='darkgreen')
    
    ax2.text(5, 0.5, 'True ball movement (straight lines)', ha='center', fontsize=10, color='green', fontweight='bold')
    
    ax2.set_xlabel('Court Width')
    ax2.set_ylabel('Court Length')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_trajectory_comparison():
    """Create a side-by-side comparison of original vs corrected trajectories"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.set_title('Tennis Ball Trajectory: Original vs Perspective-Corrected', fontsize=16, fontweight='bold')
    
    # Create tennis court outline
    court = Rectangle((0, 0), 23.77, 10.97, linewidth=3, edgecolor='white', facecolor='darkgreen', alpha=0.8)
    ax.add_patch(court)
    
    # Add court lines
    # Center line
    ax.plot([11.885, 11.885], [0, 10.97], 'w-', linewidth=2)
    # Service lines
    ax.plot([0, 23.77], [6.4, 6.4], 'w-', linewidth=2)
    ax.plot([0, 23.77], [4.57, 4.57], 'w-', linewidth=2)
    # Singles sidelines
    ax.plot([4.115, 4.115], [0, 10.97], 'w-', linewidth=2)
    ax.plot([19.655, 19.655], [0, 10.97], 'w-', linewidth=2)
    
    # Simulate original trajectory (curved due to perspective)
    x_orig = np.linspace(2, 22, 100)
    y_orig = 5.5 + 2 * np.sin(0.8 * x_orig) * np.exp(-0.05 * x_orig)
    
    # Simulate corrected trajectory (straight lines between bounces)
    bounce_x = [3, 8, 15, 21]
    bounce_y = [3, 7, 4, 8]
    
    # Plot original trajectory
    ax.plot(x_orig, y_orig, 'r-', linewidth=4, label='Original (with 3D distortion)', alpha=0.8)
    
    # Plot corrected trajectory (straight segments)
    for i in range(len(bounce_x) - 1):
        ax.plot([bounce_x[i], bounce_x[i+1]], [bounce_y[i], bounce_y[i+1]], 
                'lime', linewidth=4, alpha=0.9)
    
    # Mark bounce points
    ax.scatter(bounce_x, bounce_y, c='yellow', s=200, zorder=5, edgecolor='black', 
               linewidth=3, label='Detected Bounces')
    
    # Add legend and annotations
    ax.plot([], [], 'lime', linewidth=4, label='Corrected (straight segments)')
    ax.legend(loc='upper right', fontsize=12)
    
    # Add explanatory text
    ax.text(11.885, 9.5, 'TENNIS COURT', ha='center', fontsize=14, color='white', fontweight='bold')
    ax.text(2, 1, 'The RED line shows the distorted trajectory\nfrom 3D-to-2D projection', 
            fontsize=10, color='red', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.text(15, 1, 'The GREEN lines show the corrected\nstraight-line trajectory', 
            fontsize=10, color='darkgreen', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlim(-1, 25)
    ax.set_ylim(-1, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig

def main():
    """Create and show the visualization diagrams"""
    
    print("🎾 Creating perspective correction visualization...")
    
    # Create the conceptual diagram
    fig1 = create_perspective_diagram()
    fig1.savefig('perspective_correction_concept.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: perspective_correction_concept.png")
    
    # Create the trajectory comparison
    fig2 = create_trajectory_comparison()
    fig2.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: trajectory_comparison.png")
    
    # Show the plots
    plt.show()
    
    print("\n🎾 Perspective Correction Explanation:")
    print("=" * 50)
    print("1. The camera sees a 3D ball trajectory projected onto a 2D court")
    print("2. This creates curved paths even for straight ball movement")
    print("3. Perspective correction detects bounces and creates straight lines")
    print("4. Result: True top-down view showing actual ball movement")

if __name__ == "__main__":
    main()
