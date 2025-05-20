import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from ljhouses.pythonsims import simulate_collisions_until_no_collisions_simple
from matplotlib.colors import LinearSegmentedColormap
import os
import matplotlib.animation as animation
import pandas as pd

df = pd.read_csv("data/buildings_with_addr_count_and_KOM_and_locations_epsg23032.csv")
coord = df[['x','y']].values


#find coord center for KOM =101
kom = 101
kom_df = df[df['KOM'] == kom]
kom_coord = kom_df[['x','y']].values
kom_coord = kom_coord.mean(axis=0)
print(kom_coord)


xmin = kom_coord[0] - 4000
xmax = kom_coord[0] + 1000

ymin = kom_coord[1] - 2000
ymax = kom_coord[1] + 4000


#try some other coord
#  the ration is 213mm x 285mm
# xmin = kom_coord[0]-2500
# ymin = kom_coord[1]

xmin = 726000
ymin = 6171150

xmin = 718400 
ymin = 6176150
# xmin = 693500
# ymin = 6170000

zoom_factor = 0.35
ratio = 285/213

xmax = xmin+10000*zoom_factor

#choose y max to ensure the ratio
ymax = ymin + 10000*zoom_factor * ratio
print(f'{ratio}, {(ymax-ymin)/(xmax-xmin)}')

scale =0.6e3 * (zoom_factor/0.5)**(-1) *4
std = 1.5

# xmax = 730000

# ymin = 6171000
# ymax = 6175000

#filter coord
coord_zoom = coord[(coord[:, 0] > xmin) & (coord[:, 0] < xmax) & (coord[:, 1] > ymin) & (coord[:, 1] < ymax)]
print(len(coord_zoom))

#make them start from 0
coord_zoom[:, 0] = coord_zoom[:, 0] - xmin
coord_zoom[:, 1] = coord_zoom[:, 1] - ymin

def add_noise(coord, std,scale):
    # Create a copy of the input array to avoid modifying the original
    coord_copy = np.copy(coord)
    #coord_copy[:, 1] = coord_copy[:, 1] + min(np.exp(coord_copy[:, 1]/scale)) * np.random.normal(0, std, len(coord_copy))
    coord_copy[:, 1] = coord_copy[:, 1] + np.exp(coord_copy[:, 1]/scale) * np.random.normal(0, std, len(coord_copy))
    coord_copy[:, 0] = coord_copy[:, 0] + np.exp(coord_copy[:, 1]/scale) * np.random.normal(0, std, len(coord_copy))

    return coord_copy

cm = 1/2.54  # centimeters in inches
figsize = (21.5*cm, 28.5*cm)  # A4 size in inches


def save_plots(coord, xmin, ymin, zoom_factor, ratio=285/213, std=1,scale=1e3, output_dir="figures",save_coords_csv=False):
    """
    Create and save three plots with filenames including position parameters
    
    Parameters:
    -----------
    coord : numpy array
        Original coordinates array with shape (n, 2)
    xmin, ymin : float
        Minimum x and y coordinates for filtering
    zoom_factor : float
        Zoom factor for the plot
    ratio : float, optional
        Aspect ratio (default: 285/213 - A4 ratio)
    std : float, optional
        Noise standard deviation (default: 1)
    scale : float, optional
        Scale parameter for noise (default: 1e3)
    output_dir : str, optional
        Directory to save figures (default: "figures")
    """
    # Calculate derived parameters
    world_xmax = xmin + 10000 * zoom_factor
    world_ymax = ymin + 10000 * zoom_factor * ratio
    
    # Filter coordinates
    coord_zoom_orig = coord[(coord[:, 0] > xmin) & (coord[:, 0] < world_xmax) & 
                            (coord[:, 1] > ymin) & (coord[:, 1] < world_ymax)]
    
    if len(coord_zoom_orig) == 0:
        print(f"No points found for xmin={xmin}, ymin={ymin}, zoom_factor={zoom_factor}. Skipping plots.")
        return
        
    print(f"Number of points: {len(coord_zoom_orig)}")
    
    # Make a copy for manipulation and shift to start from 0,0
    coord_zoom = np.copy(coord_zoom_orig)
    coord_zoom[:, 0] = coord_zoom[:, 0] - xmin
    coord_zoom[:, 1] = coord_zoom[:, 1] - ymin

    # Define plot area extent after shifting to 0,0
    plot_xmax = coord_zoom[:, 0].max()
    plot_ymax = coord_zoom[:, 1].max()
    
    # Set figure size
    cm_local = 1/2.54  # centimeters in inches
    current_figsize = (21.3*cm_local, 28.5*cm_local)  # A4 size in inches
    
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"coord_x{int(xmin)}_y{int(ymin)}_z{zoom_factor:.2f}"

    # --- CSV Saving Setup ---
    coords_csv_specific_dir = None
    if save_coords_csv:
        coords_csv_basedir = "coordinates" # Main directory for all coordinate CSVs
        # Specific subdirectory for this set of x, y, zoom parameters
        coords_csv_specific_dir = os.path.join(coords_csv_basedir, f"x{int(xmin)}_y{int(ymin)}_z{zoom_factor:.2f}")
        os.makedirs(coords_csv_specific_dir, exist_ok=True)

    def _save_coordinates_to_csv(coordinates_array, plot_type_suffix):
        if save_coords_csv and coords_csv_specific_dir is not None:
            if coordinates_array.ndim == 1:
                 coordinates_array = coordinates_array.reshape(1, -1)
            
            if coordinates_array.shape[1] != 2:
                print(f"Warning: Coordinates for '{plot_type_suffix}' do not have 2 columns. Skipping CSV save.")
                return

            filepath = os.path.join(coords_csv_specific_dir, f"{plot_type_suffix}.csv")
            df_coords = pd.DataFrame(coordinates_array, columns=['x', 'y'])
            df_coords.to_csv(filepath, index=False)
            print(f"Saved coordinates for '{plot_type_suffix}' to {filepath}")

    def _create_and_save_scatter(plot_coordinates, point_color_arg, 
                                 xlim_tuple, ylim_tuple, 
                                 filename_suffix, generic_savename=None,
                                 fig_face_color='white', ax_face_color='white', 
                                 save_fig_face_color=None, point_size=2.0, s_alpha=1.0,
                                 s_edgecolors='none',
                                flip_x=False,
                                flip_y=False):
                                
        _save_coordinates_to_csv(plot_coordinates, filename_suffix)
        coords_to_plot = np.copy(plot_coordinates)

        if flip_x:
            coords_to_plot[:, 0] = -coords_to_plot[:, 0] # Reflect across y-axis
            xlim_tuple = (-xlim_tuple[1], -xlim_tuple[0])
        
        if flip_y:
            # Flip y-coordinates with respect to the y-axis limits.
            coords_to_plot[:, 1] = ylim_tuple[1] - coords_to_plot[:, 1]
            plot_ymax = ylim_tuple[1] - ylim_tuple[0]
            ylim_tuple = (-ylim_tuple[1], -ylim_tuple[0])

        
        fig = plt.figure(figsize=current_figsize, facecolor=fig_face_color)
        ax = fig.add_subplot(111, facecolor=ax_face_color)
        
        ax.scatter(plot_coordinates[:, 0], plot_coordinates[:, 1], c=point_color_arg, 
                   s=point_size, alpha=s_alpha, edgecolors=s_edgecolors)
        
        ax.axis('off')
        ax.set_xlim(xlim_tuple)
        ax.set_ylim(ylim_tuple)
        ax.set_aspect('equal', adjustable='box')
        
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        
        actual_save_facecolor = save_fig_face_color if save_fig_face_color is not None else fig_face_color
        
        specific_path = f"{output_dir}/{base_filename}_{filename_suffix}.png"
        plt.savefig(specific_path, dpi=500, bbox_inches='tight', 
                    pad_inches=0, facecolor=actual_save_facecolor)
        
        if generic_savename:
            generic_path = f"{output_dir}/{generic_savename}.png"
            plt.savefig(generic_path, dpi=500, bbox_inches='tight', 
                        pad_inches=0, facecolor=actual_save_facecolor)
        
        plt.close(fig)

    # --- Plot 1: Original coordinates ---
    _create_and_save_scatter(coord_zoom, 'k', 
                             (0, plot_xmax), (0, plot_ymax), 
                             "original", point_size=2)

    # --- Plot 2: Noise ---
    coord_zoom_noise_plot2 = add_noise(np.copy(coord_zoom), std, scale) # Use function params std, scale
    coords_for_plot2 = np.copy(coord_zoom_noise_plot2)
    coords_for_plot2[:, 0] = -coords_for_plot2[:, 0] # Flip x-coordinates for plotting
    _create_and_save_scatter(coords_for_plot2, 'k', 
                             (-plot_xmax, 0), (0, plot_ymax), 
                             "noise", point_size=2)

    # --- Plot 3: Color gradient with noise (White to Pink on Black BG) ---
    coord_zoom_noise_grad = add_noise(np.copy(coord_zoom), std, int(1.2e3)) # Specific scale for this
    
    # No noise for the bottom 1/3 - revert to original coord_zoom values
    # Ensure mask is applied to the correct (original) indices if coord_zoom was filtered
    # Assuming coord_zoom and coord_zoom_noise_grad are aligned
    mask_bottom_third = coord_zoom_noise_grad[:, 1] < (plot_ymax / 3)
    coord_zoom_noise_grad[mask_bottom_third, 0] = coord_zoom[mask_bottom_third, 0]
    coord_zoom_noise_grad[mask_bottom_third, 1] = coord_zoom[mask_bottom_third, 1]

        # Define the colormap: white for the first 40% of the range,
    # then transitioning to the second color (Greenish: R=0, G=0.85, B=0).
    colors_map_white_pink = [
        (0.0, (1, 1, 1)),      # Start color: white at position 0.0
        (0.15, (1, 1, 1)),      # White persists until position 0.4
        (1.0, (0, 0.5, 1))    # End color at position 1.0 (Greenish, not pink as original comment suggested)
    ]
    white_to_pink_cmap = LinearSegmentedColormap.from_list('white_to_pink', colors_map_white_pink)
    
    norm_y_grad = plt.Normalize(coord_zoom_noise_grad[:, 1].min(), coord_zoom_noise_grad[:, 1].max())
    point_colors_white_pink_grad = white_to_pink_cmap(norm_y_grad(coord_zoom_noise_grad[:, 1]))
    
   
    _create_and_save_scatter(coord_zoom_noise_grad, point_colors_white_pink_grad, 
                             (0,plot_xmax), (0, plot_ymax), 
                             "gradient", generic_savename="gradient",
                             fig_face_color='black', ax_face_color='black', point_size=2.5)

    # --- Collision-based plots ---
    # R calculation based on y-values of coord_zoom_noise_grad (which has ymax = plot_ymax)
    R_values = coord_zoom_noise_grad[:, 1] / plot_ymax * 10.0
    mask_R_half = coord_zoom_noise_grad[:, 1] < (plot_ymax / 2.0)
    R_values[mask_R_half] = 0.1
    # Ensure R_values are positive
    R_values[R_values <= 0] = 0.01 

    coord_zoom_collided = simulate_collisions_until_no_collisions_simple(
        np.copy(coord_zoom_noise_grad), R_values, mass_prop_to_area=True
    )
    

    # Colors are based on pre-collision y-values (coord_zoom_noise_grad)
    _create_and_save_scatter(coord_zoom_collided, point_colors_white_pink_grad, 
                             (0,plot_xmax), (0, plot_ymax), 
                             "gradient_collision", generic_savename="gradient_collision",
                             fig_face_color='black', ax_face_color='black', point_size=2.5)

    # --- Plot 5: Gradient collision white points plot (White points on Black BG) ---
    _create_and_save_scatter(coord_zoom_collided, 'w', 
                             (0,plot_xmax), (0, plot_ymax), 
                             "gradient_collision_white", generic_savename="gradient_collision_white",
                             fig_face_color='black', ax_face_color='black', point_size=2.5)
    
    # --- Plot 6: Collision plot with white background (Black points, White axes, White Fig) ---
    _create_and_save_scatter(coord_zoom_collided, 'k', #
                             (0,plot_xmax), (0, plot_ymax), 
                             "collision_black_on_white", generic_savename="collision_black_on_white",
                             fig_face_color='white', ax_face_color='white', 
                             save_fig_face_color='white', point_size=2.5)

    print(f"Saved plots for base: {output_dir}/{base_filename}")

    if save_coords_csv and coords_csv_specific_dir:
        print(f"Saved coordinates to directory: {coords_csv_specific_dir}")





def create_animation_sequence(coord, xmin, ymin, zoom_factor, 
                              ratio=285/213, std=1, scale=1e3, 
                              output_dir="animations", base_animation_name="animation_sequence",
                              frames_per_transition=50, fps=20, point_size=2.5,
                              fig_face_color='black', ax_face_color='black'):
    """
    Creates an MP4 video animation showing transitions:
    1. Original to Noisy positions (colors interpolate from white to final gradient).
    2. Noisy to Collided positions (colors remain final gradient).
    3. Collided to Noisy positions (colors remain final gradient).
    4. Noisy to Original positions (colors interpolate from final gradient to white).
    The animation loops smoothly (ping-pong).

    Parameters:
    -----------
    coord, xmin, ymin, zoom_factor, ratio, std, scale, output_dir: Same as save_plots.
    base_animation_name : str
        Base name for the output MP4 file.
    frames_per_transition : int
        Number of frames for each of the 4 transition stages.
    fps : int
        Frames per second for the animation.
    point_size : float
        Size of the scatter points.
    fig_face_color, ax_face_color : str
        Background colors for the figure and axes.
    """
    print(f"Starting ping-pong MP4 animation generation for x{int(xmin)}_y{int(ymin)}_z{zoom_factor:.2f}")

    # --- 1. Data Preparation (similar to save_plots) ---
    world_xmax = xmin + 10000 * zoom_factor
    world_ymax = ymin + 10000 * zoom_factor * ratio
    
    coord_zoom_orig_filter = coord[(coord[:, 0] > xmin) & (coord[:, 0] < world_xmax) & 
                                   (coord[:, 1] > ymin) & (coord[:, 1] < world_ymax)]
    
    if len(coord_zoom_orig_filter) == 0:
        print(f"No points found for animation. Skipping.")
        return

    # Stage 1: Original Coordinates (shifted to 0,0)
    coord_s1_original = np.copy(coord_zoom_orig_filter)
    coord_s1_original[:, 0] -= xmin
    coord_s1_original[:, 1] -= ymin

    plot_xmax = coord_s1_original[:, 0].max() if len(coord_s1_original) > 0 else 1.0
    plot_ymax = coord_s1_original[:, 1].max() if len(coord_s1_original) > 0 else 1.0

    # Stage 2: Noisy Coordinates
    coord_s2_noisy = add_noise(np.copy(coord_s1_original), std, int(1.2e3)) 
    mask_bottom_third = coord_s2_noisy[:, 1] < (plot_ymax / 3)
    coord_s2_noisy[mask_bottom_third, 0] = coord_s1_original[mask_bottom_third, 0]
    coord_s2_noisy[mask_bottom_third, 1] = coord_s1_original[mask_bottom_third, 1]

    # Stage 3: Collided Coordinates
    R_values = coord_s2_noisy[:, 1] / plot_ymax * 10.0
    mask_R_half = coord_s2_noisy[:, 1] < (plot_ymax / 2.0)
    R_values[mask_R_half] = 0.1
    R_values[R_values <= 0] = 0.01 
    coord_s3_collided = simulate_collisions_until_no_collisions_simple(
        np.copy(coord_s2_noisy), R_values, mass_prop_to_area=True
    )

    # --- 2. Color Preparation ---
    num_points = len(coord_s1_original)
    colors_initial = np.ones((num_points, 3))  # White

    colors_map_def = [
        (0.0, (1, 1, 1)),      
        (0.15, (1, 1, 1)),     
        (1.0, (0, 0.5, 1))     
    ]
    final_cmap = LinearSegmentedColormap.from_list('final_gradient_anim', colors_map_def)
    norm_y_final = plt.Normalize(coord_s2_noisy[:, 1].min(), coord_s2_noisy[:, 1].max())
    colors_final_rgba = final_cmap(norm_y_final(coord_s2_noisy[:, 1]))
    colors_final = colors_final_rgba[:, :3] 

    # --- 3. Animation Setup ---
    cm_local = 1/2.54
    current_figsize = (21.3*cm_local, 28.5*cm_local)
    
    fig = plt.figure(figsize=current_figsize, facecolor=fig_face_color)
    ax = fig.add_subplot(111, facecolor=ax_face_color)
    ax.axis('off')
    ax.set_xlim(0, plot_xmax)
    ax.set_ylim(0, plot_ymax)
    ax.set_aspect('equal', adjustable='box')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    scatter = ax.scatter(coord_s1_original[:, 0], coord_s1_original[:, 1], 
                         s=point_size, c=colors_initial, edgecolors='none')

    total_frames = 4 * frames_per_transition 

    def update(frame):
        current_coords_x, current_coords_y = None, None
        current_colors = None
        
        stage_frame = frame % frames_per_transition
        current_stage = frame // frames_per_transition
        
        alpha = stage_frame / (frames_per_transition - 1) if frames_per_transition > 1 else 1.0

        if current_stage == 0: # Stage 0: Original to Noisy
            current_coords_x = (1 - alpha) * coord_s1_original[:, 0] + alpha * coord_s2_noisy[:, 0]
            current_coords_y = (1 - alpha) * coord_s1_original[:, 1] + alpha * coord_s2_noisy[:, 1]
            current_colors_r = (1 - alpha) * colors_initial[:, 0] + alpha * colors_final[:, 0]
            current_colors_g = (1 - alpha) * colors_initial[:, 1] + alpha * colors_final[:, 1]
            current_colors_b = (1 - alpha) * colors_initial[:, 2] + alpha * colors_final[:, 2]
            current_colors = np.vstack((current_colors_r, current_colors_g, current_colors_b)).T
        elif current_stage == 1: # Stage 1: Noisy to Collided
            current_coords_x = (1 - alpha) * coord_s2_noisy[:, 0] + alpha * coord_s3_collided[:, 0]
            current_coords_y = (1 - alpha) * coord_s2_noisy[:, 1] + alpha * coord_s3_collided[:, 1]
            current_colors = colors_final
        elif current_stage == 2: # Stage 2: Collided to Noisy
            current_coords_x = (1 - alpha) * coord_s3_collided[:, 0] + alpha * coord_s2_noisy[:, 0]
            current_coords_y = (1 - alpha) * coord_s3_collided[:, 1] + alpha * coord_s2_noisy[:, 1]
            current_colors = colors_final
        elif current_stage == 3: # Stage 3: Noisy to Original
            current_coords_x = (1 - alpha) * coord_s2_noisy[:, 0] + alpha * coord_s1_original[:, 0]
            current_coords_y = (1 - alpha) * coord_s2_noisy[:, 1] + alpha * coord_s1_original[:, 1]
            current_colors_r = (1 - alpha) * colors_final[:, 0] + alpha * colors_initial[:, 0]
            current_colors_g = (1 - alpha) * colors_final[:, 1] + alpha * colors_initial[:, 1]
            current_colors_b = (1 - alpha) * colors_final[:, 2] + alpha * colors_initial[:, 2]
            current_colors = np.vstack((current_colors_r, current_colors_g, current_colors_b)).T
        
        scatter.set_offsets(np.c_[current_coords_x, current_coords_y])
        scatter.set_facecolor(current_colors)
        
        if frame % (frames_per_transition // 2 if frames_per_transition > 1 else 1) == 0 :
            print(f"Processing animation frame {frame+1}/{total_frames} (Stage {current_stage+1})")
        return scatter,

    # --- 4. Create and Save Animation ---
    os.makedirs(output_dir, exist_ok=True)
    animation_filename_base = f"{base_animation_name}_pingpong_x{int(xmin)}_y{int(ymin)}_z{zoom_factor:.2f}"
    # Changed extension to .mp4
    animation_filepath = os.path.join(output_dir, f"{animation_filename_base}.mp4")

    try:
        # Ensure Matplotlib knows where to find ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg' # Or full path if not in system PATH

        # Explicitly create an FFMpegWriter instance
        #ffmpeg_writer = animation.FFMpegWriter(fps=fps, codec='libx264', bitrate=-1)
        ffmpeg_writer = animation.FFMpegWriter(fps=fps, codec='libopenh264', bitrate=-1)
        # 'libx264' is a good default codec for MP4.
        # bitrate=-1 will use a variable bitrate for good quality. Adjust if needed.

        ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                      interval=1000/fps, blit=True)
        
        print(f"Saving ping-pong MP4 to {animation_filepath} (this may take a while)...")
        # Use the explicit ffmpeg_writer
        ani.save(animation_filepath, writer=ffmpeg_writer, dpi=200) # dpi can be adjusted
        print("Ping-pong MP4 saved successfully.")
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg and ensure it's in your system PATH.")
        print("You can typically install it via your system's package manager (e.g., 'sudo apt install ffmpeg')")
    except RuntimeError as e:
        print(f"A RuntimeError occurred during animation saving (often ffmpeg related): {e}")
        print("Check your ffmpeg installation, version, and if it supports the chosen codec (e.g., libx264).")
        print("Try installing/reinstalling ffmpeg with common codecs: 'sudo apt install ffmpeg libavcodec-extra'")
    except Exception as e:
        print(f"An unexpected error occurred during animation saving: {e}")
    finally:
        plt.close(fig)




save_plots(coord, xmin, ymin, zoom_factor,std=std,scale=scale,save_coords_csv=True)
# create_animation_sequence(coord, xmin, ymin, zoom_factor, std=std, scale=scale, frames_per_transition=30, fps=30)