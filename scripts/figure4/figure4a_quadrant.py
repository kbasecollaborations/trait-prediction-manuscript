import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure with a clean, publication-ready style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Define colors
concordant_color = '#c8e6c9'  # Light green for concordant
discordant_fp_color = '#ffcdd2'  # Light red for false positive
discordant_fn_color = '#fff9c4'  # Light yellow for false negative
concordant_tn_color = '#e3f2fd'  # Light blue for true negative

# Draw the four quadrants
# Top-left: GapMind+ / Experiment+ (True Positive - Concordant)
quad_tl = FancyBboxPatch((0.5, 5.2), 4.3, 4.3, 
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=concordant_color, edgecolor='#388e3c', linewidth=2)
ax.add_patch(quad_tl)

# Top-right: GapMind- / Experiment+ (False Negative - Discordant)
quad_tr = FancyBboxPatch((5.2, 5.2), 4.3, 4.3,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=discordant_fn_color, edgecolor='#f57c00', linewidth=2)
ax.add_patch(quad_tr)

# Bottom-left: GapMind+ / Experiment- (False Positive - Discordant)
quad_bl = FancyBboxPatch((0.5, 0.5), 4.3, 4.3,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=discordant_fp_color, edgecolor='#d32f2f', linewidth=2)
ax.add_patch(quad_bl)

# Bottom-right: GapMind- / Experiment- (True Negative - Concordant)
quad_br = FancyBboxPatch((5.2, 0.5), 4.3, 4.3,
                          boxstyle="round,pad=0.05,rounding_size=0.2",
                          facecolor=concordant_tn_color, edgecolor='#1976d2', linewidth=2)
ax.add_patch(quad_br)

# Add axis labels
ax.text(5, 10.2, 'GapMind Prediction', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.text(2.65, 9.7, 'Pathway Present (+)', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2e7d32')
ax.text(7.35, 9.7, 'Pathway Absent (−)', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#c62828')

ax.text(-0.3, 5, 'Experimental\nOutcome', ha='center', va='center', fontsize=14, fontweight='bold', rotation=90)
ax.text(0.15, 7.35, 'Growth (+)', ha='center', va='center', fontsize=12, fontweight='bold', color='#2e7d32', rotation=90)
ax.text(0.15, 2.65, 'No Growth (−)', ha='center', va='center', fontsize=12, fontweight='bold', color='#c62828', rotation=90)

# Top-left quadrant content (True Positive - Concordant)
ax.text(2.65, 9.1, 'CONCORDANT', ha='center', va='center', fontsize=11, fontweight='bold', color='#1b5e20')
ax.text(2.65, 8.65, 'True Positive', ha='center', va='center', fontsize=10, fontstyle='italic', color='#2e7d32')
ax.text(2.65, 7.9, '✓ Genes present & functional', ha='center', va='center', fontsize=9, color='#1b5e20')
ax.text(2.65, 7.5, '✓ Pathway correctly annotated', ha='center', va='center', fontsize=9, color='#1b5e20')
ax.text(2.65, 7.1, '✓ Experiment accurately measured', ha='center', va='center', fontsize=9, color='#1b5e20')
ax.text(2.65, 6.4, 'High-quality training samples', ha='center', va='center', fontsize=9, fontweight='bold', 
        color='#1b5e20', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#388e3c', alpha=0.8))

# Top-right quadrant content (False Negative - Discordant: GapMind-, Experiment+)
ax.text(7.35, 9.1, 'DISCORDANT', ha='center', va='center', fontsize=11, fontweight='bold', color='#e65100')
ax.text(7.35, 8.65, 'False Negative', ha='center', va='center', fontsize=10, fontstyle='italic', color='#f57c00')

# Causes for FN (genes absent but growth observed)
ax.text(7.35, 8.1, 'Possible Causes:', ha='center', va='center', fontsize=9, fontweight='bold', color='#e65100')
ax.text(5.45, 7.65, '• Annotation: homolog not detected', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(5.45, 7.3, '• Biology: alternative pathway', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(5.45, 6.95, '• Biology: promiscuous enzymes', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(5.45, 6.6, '• Measurement: false positive', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(5.45, 6.25, '• Media: growth on base media', ha='left', va='center', fontsize=8.5, color='#424242')

ax.text(7.35, 5.6, 'Growth via unknown\nmechanism', ha='center', va='center', fontsize=9, fontweight='bold',
        color='#e65100', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#f57c00', alpha=0.8))

# Bottom-left quadrant content (False Positive - Discordant: GapMind+, Experiment-)
ax.text(2.65, 4.4, 'DISCORDANT', ha='center', va='center', fontsize=11, fontweight='bold', color='#b71c1c')
ax.text(2.65, 3.95, 'False Positive', ha='center', va='center', fontsize=10, fontstyle='italic', color='#c62828')

# Causes for FP (genes present but no growth)
ax.text(2.65, 3.4, 'Possible Causes:', ha='center', va='center', fontsize=9, fontweight='bold', color='#b71c1c')
ax.text(0.75, 2.95, '• Regulation: pathway repressed', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(0.75, 2.6, '• Measurement: insufficient time', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(0.75, 2.25, '• Media: missing cofactors', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(0.75, 1.9, '• Annotation: false positive call', ha='left', va='center', fontsize=8.5, color='#424242')
ax.text(0.75, 1.55, '• Biology: incomplete pathway', ha='left', va='center', fontsize=8.5, color='#424242')

ax.text(2.65, 0.9, 'Genes present but\nnot functional', ha='center', va='center', fontsize=9, fontweight='bold',
        color='#b71c1c', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#d32f2f', alpha=0.8))

# Bottom-right quadrant content (True Negative - Concordant)
ax.text(7.35, 4.4, 'CONCORDANT', ha='center', va='center', fontsize=11, fontweight='bold', color='#0d47a1')
ax.text(7.35, 3.95, 'True Negative', ha='center', va='center', fontsize=10, fontstyle='italic', color='#1976d2')
ax.text(7.35, 3.2, '✓ Genes absent', ha='center', va='center', fontsize=9, color='#0d47a1')
ax.text(7.35, 2.8, '✓ No alternative pathway', ha='center', va='center', fontsize=9, color='#0d47a1')
ax.text(7.35, 2.4, '✓ Experiment accurately measured', ha='center', va='center', fontsize=9, color='#0d47a1')
ax.text(7.35, 1.7, 'High-quality training samples', ha='center', va='center', fontsize=9, fontweight='bold',
        color='#0d47a1', bbox=dict(boxstyle='round', facecolor='white', edgecolor='#1976d2', alpha=0.8))

# Add a legend/key at the bottom
legend_y = -0.5
ax.text(2.5, legend_y, '■ Concordant (use for training)', ha='center', va='center', fontsize=10, 
        color='#1b5e20', fontweight='bold')
ax.text(7.5, legend_y, '■ Discordant (analyze for data quality insights)', ha='center', va='center', fontsize=10,
        color='#b71c1c', fontweight='bold')

# Add small colored squares for legend
ax.add_patch(mpatches.Rectangle((1.3, legend_y - 0.15), 0.3, 0.3, facecolor=concordant_color, edgecolor='#388e3c', linewidth=1))
ax.add_patch(mpatches.Rectangle((5.9, legend_y - 0.15), 0.3, 0.3, facecolor=discordant_fp_color, edgecolor='#d32f2f', linewidth=1))

plt.tight_layout()
plt.savefig('/home/claude/figure6a_quadrant.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.savefig('/home/claude/figure6a_quadrant.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)
print("Figure saved successfully!")
