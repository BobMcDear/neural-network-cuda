#ifndef MODULE_H
#define MODULE_H

#define block_size 32


class Module{
    public:
        float *inp, *out;
        int sz_out;
        
        virtual void forward(float *inp, float *out){};
        virtual void backward(){};
        virtual void update(){};
};


#endif