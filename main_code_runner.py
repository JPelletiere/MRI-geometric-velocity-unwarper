from calc_pixel_loc import calc_pixel_loc_main
from geometric_unwarp import geometric_unwarp_main
from velocity_correction import velocity_unwarping_main
from bfields import bfield_main

run_bfields = False # This is a very heavy computation that can take over an hour, run this BEFORE needed for on the spot analysis
run_calc_pixel_loc = True
run_gradient_unwarp = False
run_MainDataProcessing = True

def main() :
    if run_bfields:
        print("Calculating B-fields")
        bfield_main()
        print('B-field generation complete')

    if run_calc_pixel_loc:
        print("Running calc_pixel_loc...")
        calc_pixel_loc_main()
        print("calc_pixel_loc completed.\n")

    if run_gradient_unwarp:
        print("Unwarping geometry...")
        geometric_unwarp_main()
        print("geometric unwarping completed.\n")

    if run_MainDataProcessing:
        print("Unwarping velocity...")
        velocity_unwarping_main()
        print("velocity unwarping completed.\n")
        

if __name__ == "__main__":
    main()
