using Unitful
using Unitful: µM, M, nm, µm, s, μs, Na, L, 𝐍


"""Converts a rate constant from 1/(uM*s) to nm^3/us)"""
function convert_to_microrate(macrorate::Float64)
    return macrorate / 0.602214076
end

"""Converts a rate constant from nm^3/us) to 1/(uM*s)"""
function convert_to_macrorate(microrate::Float64)
    return microrate * 0.602214076
end

function concentration_to_copy_number(concentration_uM::Float64, volume_um3::Float64)
    # Convert concentration from uM to M
    concentration_M = concentration_uM * 1e-6
    # Convert volume from um^3 to L
    volume_L = volume_um3 * 1e-15
    # Calculate the number of moles in the solution
    moles = concentration_M * volume_L
    # Convert moles to molecules (i.e., copy number) using Avogadro's number
    copy_number = moles * 6.022e23
    return convert(Int, round(copy_number))
end


"""Calculates the NERDSS waterbox size, when total volume and DF are known"""
function calculate_waterbox(fixedvolume::Float64, DF::Float64)
    nanofixedvolume = fixedvolume*1e9 #volume in nm^3 from μm^3
    area = nanofixedvolume / DF #volume in nm^3 divided by DF in nm

    x = y = round(√area) #square nm of the floor
    z = round(DF) #heigh of the waterbox in nm

    @info "WaterBox is [$x, $y, $z]"

    return [x,y,z]
end

# @unit copies "copies" Copynumber (6.022e2)*(μM*μm^3) false

# copies(1.0μM*μm^3)

# uconvert(copies, 1.3μM*μm^3)

# uconvert(µM, 7.35mM)

# uconvert(nm^3/μs, 5.1μM^-1*s^-1)

# uconvert(L, 100nm^3)



# concentration_to_copy_number(1.0, 1.0)