using Random
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
    # Get list of all .txt files in the folder
    folder_path = joinpath(@__DIR__, "../lib/maurergroup/")
    files = readdir(folder_path)

    # Pick a random index
    random_index = rand(1:length(files))

    # Get the corresponding file
    selected_file = files[random_index]
    file_path = joinpath(folder_path, selected_file)
    # Read and print the contents
    name = splitext(selected_file)[1]
    println(read(file_path, String))
    println()
    println("You've caught ", name,"!")
    println()
    println("Gotta catch them all! ☉")
end
⊚