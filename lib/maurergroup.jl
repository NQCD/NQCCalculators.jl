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
    weights = [0.25, 0.25, 0.25, 0.25]
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

