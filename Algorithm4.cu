for x = 0 to nxs do 
    for sf = 0 <= nSpat_filters do  
        for tf = 0 <= nTemp_filters do
            for fr = 0 <= nFrames - L do
                frame = getframes(R_x,x,sf,tf,fr)
                for all p = pixel in frame do
                    //third order in primary direction
                    //second order in orthogonal and temporal directions
                    I_theta = TaylorTruncation(R_x)
                end for
            end for
        end for
    end for
end for