using Random
using StatsBase
#using FileSystem

function get_Lukas()
    file_path = joinpath(@__DIR__, "../lib/maurergroup/", "Lukas.txt")
    println(read(file_path, String))
end

function bold_unicode(str)
    bold_str = ""
    for c in str
        if 'A' <= c <= 'Z'
            bold_str *= Char(Int(c) - Int('A') + 0x1D400)
        elseif 'a' <= c <= 'z'
            bold_str *= Char(Int(c) - Int('a') + 0x1D41A)
        else
            bold_str *= c
        end
    end
    return bold_str
end

function get_maurergroup()
    weights = [ # will change this to read in a file with the values laid out so will be easier to edit
        0.115075965, # Alex
        0.134255292, # Ash
        0.014384496, # Chen
        0.038358655, # Connor
        0.028768991, # Danjo
        0.095896637, # Duck
        0.009589664, # Dylan
        0.095896637, # Henry Snowden
        0.009589664, # Henry Thake
        0.057537982, # Lukas
        0.134255292, # Nils
        0.100691469, # Reini
        0.115075965e-2, # Shiny Alex
        0.134255292e-2, # Shiny Ash
        0.014384496e-2, # Shiny Chen
        0.038358655e-2, # Shiny Connor
        0.028768991e-2, # Shiny Danjo
        0.095896637e-2, # Shiny Duck
        0.009589664e-2, # Shiny Dylan
        0.095896637e-2, # Shiny Henry Snowden
        0.009589664e-2, # Shiny Henry Thake
        0.057537982e-2, # Shiny Lukas
        0.134255292e-2, # Shiny Nils
        0.100691469e-2, # Shiny Reini
        0.019371121e-2, # Shiny Wojciech
        0.015822945e-2, # Shiny Zsuzsanna
        0.019371121, # Wojciech
        0.015822945  # Zsuzsanna
    ]

    folder_path = joinpath(@__DIR__, "../lib/maurergroup/")
    # Get list of all .txt files in the folder
    files = readdir(folder_path)

    # Select a random index using the provided weights and get the corresponding file

    selected_file = sample(files, Weights(weights))

    file_path = joinpath(folder_path, selected_file)
    # Read and print the contents
    name = splitext(selected_file)[1]
    println(read(file_path, String))
    println()
    println("You've caught ", name,"!")
    println()
    println("Gotta catch them all! ☉")
end

