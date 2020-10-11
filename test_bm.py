import numpy as np
import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk
import cv2 as cv

base_w = 672
base_h = 480

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(fill=BOTH, expand=True)
        self.create_widgets()
        self.f_main = ''
        self.f_aux = ''
        self.max_disp = 32
        self.win_size = 9
        self.real_max_disp = 0
        self.trace_started = False

    def create_widgets(self):
        self.shift = tk.StringVar(value="current pixel shift is 0")
        self.info = tk.Label(self,
                             textvariable=self.shift,
                             bg='black',
                             fg='#00CC7B',
                             font="Century\ Schoolbook 16 bold")
        self.info.pack(fill=X)

        self.bar_val = tk.StringVar()
        self.bar = tk.Scale(self,
                            font="Century\ Schoolbook 12 bold",
                            label=r'set the input pixel shift',
                            from_=0, to=22,
                            #length=400,
                            tickinterval=2,
                            resolution=1, show=0,
                            orient=tk.HORIZONTAL,
                            variable=self.bar_val,
                            command=self.bar_listener)
        self.bar.pack(fill=X)

        self.frame_input = Frame(self, relief=RAISED, borderwidth=1)
        self.frame_input.pack(fill=BOTH, expand=True, side='left')

        self.im_main = Image.fromarray(np.zeros([base_h, base_w], dtype=np.uint8))
        self.im_main_r = self.im_main.resize((336, 240))
        self.tk_im_main = ImageTk.PhotoImage(self.im_main_r)
        self.canvas_main = tk.Label(self.frame_input, image=self.tk_im_main)
        self.canvas_main.pack(side='top')

        self.caption_main = tk.Label(self.frame_input, text='Main View', font="Century\ Schoolbook 12 bold",)
        self.caption_main.pack(side='top')

        self.im_aux = Image.fromarray(np.zeros([base_h, base_w], dtype=np.uint8))
        self.im_aux_r = self.im_aux.resize((336, 240))
        self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
        self.canvas_aux = tk.Label(self.frame_input, image=self.tk_im_aux)
        self.canvas_aux.pack(side='top')

        self.caption_aux = tk.Label(self.frame_input, text='Aux View', font="Century\ Schoolbook 12 bold", )
        self.caption_aux.pack(side='top')

        self.frame_output = Frame(self, relief=RAISED, borderwidth=1)
        self.frame_output.pack(fill=BOTH, expand=True, side='right')

        self.im_disp_bm = Image.fromarray(np.zeros([base_h, base_w], dtype=np.uint8))
        self.im_disp_bm_r = self.im_disp_bm.resize((336, 240))
        print(self.im_disp_bm.size)
        self.tk_im_disp_bm = ImageTk.PhotoImage(self.im_disp_bm_r)
        self.canvas_disp_bm = tk.Label(self.frame_output, image=self.tk_im_disp_bm)
        self.canvas_disp_bm.pack(side='top')

        self.caption_bm = tk.Label(self.frame_output, text='BM Disparity', font="Century\ Schoolbook 12 bold", )
        self.caption_bm.pack(side='top')

        self.im_disp_sgbm = Image.fromarray(np.zeros([base_h, base_w], dtype=np.uint8))
        self.im_disp_sgbm_r = self.im_disp_sgbm.resize((336, 240))
        self.tk_im_disp_sgbm = ImageTk.PhotoImage(self.im_disp_sgbm_r)
        self.canvas_disp_sgbm = tk.Label(self.frame_output, image=self.tk_im_disp_sgbm)
        self.canvas_disp_sgbm.pack(side='top')

        self.caption_bm = tk.Label(self.frame_output, text='SGBM Disparity', font="Century\ Schoolbook 12 bold", )
        self.caption_bm.pack(side='top')

        self.selc_view = tk.Button(self,
                                   font="Century\ Schoolbook 12 bold",
                                   text=" Select A View ",
                                   bg="brown",
                                   fg='white',
                                   command=self.select_main)
        self.selc_view.pack(fill=X, pady=8)

        self.calc_bm = tk.Button(self,
                              font="Century\ Schoolbook 12 bold",
                              text=" Disparity BM ",
                              bg="green",
                              fg='white',
                              command=self.calc_disp_bm)
        self.calc_bm.pack(fill=X, pady=8)

        self.calc_sgbm = tk.Button(self,
                                 font="Century\ Schoolbook 12 bold",
                                 text=" Disparity SGBM ",
                                 bg="green",
                                 fg='white',
                                 command=self.calc_disp_sgbm)
        self.calc_sgbm.pack(fill=X, pady=8)

        self.calc_all = tk.Button(self,
                                   font="Century\ Schoolbook 12 bold",
                                   text=" Update All ",
                                   bg="green",
                                   fg='white',
                                   command=self.calc_disp_all)
        self.calc_all.pack(fill=X, pady=8)

        self.auto_play = tk.Button(self,
                                  font="Century\ Schoolbook 12 bold",
                                  text=" Auto Play ",
                                  bg="green",
                                  fg='white',
                                  command=self.auto_scroll)
        self.auto_play.pack(fill=X, pady=8)

        self.trace = tk.Button(self,
                                   font="Century\ Schoolbook 12 bold",
                                   text=" Trace Disparity ",
                                   bg="green",
                                   fg='white',
                                   command=self.trace_disp)
        self.trace.pack(fill=X, pady=8)

        self.auto_trace = tk.Button(self,
                               font="Century\ Schoolbook 12 bold",
                               text=" Auto Trace ",
                               bg="green",
                               fg='white',
                               command=self.auto_trace_disp)
        self.auto_trace.pack(fill=X, pady=8)

        self.fullscreen = tk.Button(self,
                                    font="Century\ Schoolbook 12 bold",
                                    text=" Full Screen ",
                                    bg="#003377",
                                    fg='white',
                                    command=self.fullscreen)
        self.fullscreen.pack(fill=X, pady=8)

        self.quit = tk.Button(self,
                              font="Century\ Schoolbook 12 bold",
                              text=" Exit ",
                              bg="red",
                              fg='white',
                              command=self.master.destroy)
        self.quit.pack(fill=X, pady=8)

    def select_main(self):
        print('selecting main view')
        f_ = askopenfilename()
        if f_[-8:-4] == 'main':
            self.f_main = f_
            self.f_aux = f_[:-8] + 'aux.png'
            self.real_max_disp = int(f_[-10:-8])
            self.bar['to'] = self.max_disp - 1 - self.real_max_disp
        elif f_[-7:-4] == 'aux':
            self.f_aux = f_
            self.f_main = f_[:-7] + 'main.png'
            self.real_max_disp = int(f_[-9:-7])
            self.bar['to'] = self.max_disp - 1 - self.real_max_disp
        else:
            return
        # load main
        self.im_main = Image.open(self.f_main)
        if self.master.attributes("-fullscreen"):
            self.im_main_r = self.im_main.resize((588, 420))
        else:
            self.im_main_r = self.im_main.resize((336, 240))
        self.tk_im_main = ImageTk.PhotoImage(self.im_main_r)
        self.canvas_main['image'] = self.tk_im_main
        # load aux
        self.im_aux = Image.open(self.f_aux)
        if self.master.attributes("-fullscreen"):
            self.im_aux_r = self.im_aux.resize((588, 420))
        else:
            self.im_aux_r = self.im_aux.resize((336, 240))
        self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
        self.canvas_aux['image'] = self.tk_im_aux
        # calculate the disparity map
        self.calc_disp_bm()
        self.calc_disp_sgbm()
        self.trace_started = False

    def calc_disp_bm(self, update_ui=True):
        print("calculating BM disparity map")
        dx = -int(self.bar_val.get())
        main_ = self.im_main
        aux_ = self.im_aux.rotate(0, translate=(dx, 0))
        im_L = np.array(main_)
        im_R = np.array(aux_)
        self.max_disp = 32
        self.win_size = 9
        stereo = cv.StereoBM_create(numDisparities=self.max_disp, blockSize=self.win_size)
        stereo.setUniquenessRatio(30)
        disp_s16 = stereo.compute(im_L, im_R)
        disp_s16 = np.maximum(disp_s16, 0)
        disp_f32 = np.array(disp_s16, dtype=np.float32)
        disp_u8 = np.array(disp_f32 * 255.0 / (16.0*self.max_disp), dtype=np.uint8)

        disp_u8 = np.minimum(disp_u8, (self.real_max_disp - dx) * 255 / self.max_disp)
        self.im_disp_bm = Image.fromarray(disp_u8)

        if self.master.attributes("-fullscreen"):
            self.im_aux_r = aux_.resize((588, 420))
        else:
            self.im_aux_r = aux_.resize((336, 240))
        self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
        self.canvas_aux['image'] = self.tk_im_aux

        if update_ui:
            if self.master.attributes("-fullscreen"):
                self.im_aux_r = aux_.resize((588, 420))
                self.im_disp_bm_r = self.im_disp_bm.resize((588, 420))
            else:
                self.im_aux_r = aux_.resize((336, 240))
                self.im_disp_bm_r = self.im_disp_bm.resize((336, 240))
            self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
            self.canvas_aux['image'] = self.tk_im_aux
            self.tk_im_disp_bm = ImageTk.PhotoImage(self.im_disp_bm_r)
            self.canvas_disp_bm['image'] = self.tk_im_disp_bm

    def calc_disp_sgbm(self, update_ui=True):
        print("calculating SGBM disparity map")
        dx = -int(self.bar_val.get())
        main_ = self.im_main
        aux_ = self.im_aux.rotate(0, translate=(dx, 0))
        im_L = np.array(main_)
        im_R = np.array(aux_)
        self.max_disp = 32
        self.win_size = 9
        stereo = cv.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.max_disp,
            blockSize=self.win_size,
            P1=24*self.win_size*self.win_size,
            P2=96*self.win_size*self.win_size,
            uniquenessRatio=30,
            preFilterCap=63,
            mode=3)
        disp_s16 = stereo.compute(im_L, im_R)
        disp_s16 = np.maximum(disp_s16, 0)
        disp_f32 = np.array(disp_s16, dtype=np.float32)
        disp_u8 = np.array(disp_f32 * 255.0 / (16.0*self.max_disp), dtype=np.uint8)
        disp_u8 = np.minimum(disp_u8, (self.real_max_disp - dx) * 255 / self.max_disp)
        self.im_disp_sgbm = Image.fromarray(disp_u8)

        if self.master.attributes("-fullscreen"):
            self.im_aux_r = aux_.resize((588, 420))
        else:
            self.im_aux_r = aux_.resize((336, 240))
        self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
        self.canvas_aux['image'] = self.tk_im_aux

        if update_ui:
            if self.master.attributes("-fullscreen"):
                self.im_aux_r = aux_.resize((588, 420))
                self.im_disp_sgbm_r = self.im_disp_sgbm.resize((588, 420))
            else:
                self.im_aux_r = aux_.resize((336, 240))
                self.im_disp_sgbm_r = self.im_disp_sgbm.resize((336, 240))
            self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
            self.canvas_aux['image'] = self.tk_im_aux
            self.tk_im_disp_sgbm = ImageTk.PhotoImage(self.im_disp_sgbm_r)
            self.canvas_disp_sgbm['image'] = self.tk_im_disp_sgbm

    def calc_disp_all(self):
        self.calc_disp_bm()
        self.calc_disp_sgbm()

    def fullscreen(self):
        flag = 1 - self.master.attributes("-fullscreen")
        self.master.attributes("-fullscreen", flag)
        if flag:
            self.bar['tickinterval'] = 1
            self.im_main_r = self.im_main.resize((588, 420))
            self.tk_im_main = ImageTk.PhotoImage(self.im_main_r)
            self.canvas_main['image'] = self.tk_im_main

            self.im_aux_r = self.im_aux.resize((588, 420))
            self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
            self.canvas_aux['image'] = self.tk_im_aux

            self.im_disp_bm_r = self.im_disp_bm.resize((588, 420))
            self.tk_im_disp_bm = ImageTk.PhotoImage(self.im_disp_bm_r)
            self.canvas_disp_bm['image'] = self.tk_im_disp_bm

            self.im_disp_sgbm_r = self.im_disp_sgbm.resize((588, 420))
            self.tk_im_disp_sgbm = ImageTk.PhotoImage(self.im_disp_sgbm_r)
            self.canvas_disp_sgbm['image'] = self.tk_im_disp_sgbm
        else:
            self.bar['tickinterval'] = 2
            self.im_main_r = self.im_main.resize((336, 240))
            self.tk_im_main = ImageTk.PhotoImage(self.im_main_r)
            self.canvas_main['image'] = self.tk_im_main

            self.im_aux_r = self.im_aux.resize((336, 240))
            self.tk_im_aux = ImageTk.PhotoImage(self.im_aux_r)
            self.canvas_aux['image'] = self.tk_im_aux

            self.im_disp_bm_r = self.im_disp_bm.resize((336, 240))
            self.tk_im_disp_bm = ImageTk.PhotoImage(self.im_disp_bm_r)
            self.canvas_disp_bm['image'] = self.tk_im_disp_bm

            self.im_disp_sgbm_r = self.im_disp_sgbm.resize((336, 240))
            self.tk_im_disp_sgbm = ImageTk.PhotoImage(self.im_disp_sgbm_r)
            self.canvas_disp_sgbm['image'] = self.tk_im_disp_sgbm

    def bar_listener(self, val):
        self.shift.set('current pixel shift is %s' % val)

    def auto_scroll(self):
        for i in range(int(self.bar['to'])+1):
            self.bar.set(i)
            self.bar.update()
            self.calc_disp_all()

    def compare_disp(self, calc_delta):
        if calc_delta:
            # save current states of disparity
            self.last_disp_bm = np.array(self.im_disp_bm) + (256 / self.max_disp)
            self.last_disp_sgbm = np.array(self.im_disp_sgbm) + (256 / self.max_disp)
        # calculate the current disparity maps
        self.calc_disp_bm(False)
        self.calc_disp_sgbm(False)
        if calc_delta:
            self.bm_trace_sum = self.bm_trace_sum + np.abs((np.array(self.im_disp_bm) - self.last_disp_bm))
            self.sgbm_trace_sum = self.sgbm_trace_sum + np.abs((np.array(self.im_disp_sgbm) - self.last_disp_sgbm))
        else:
            self.bm_trace_sum = np.zeros([base_h, base_w], dtype=np.int32)
            self.sgbm_trace_sum = np.zeros([base_h, base_w], dtype=np.int32)
            self.im_bm_trace_mean = Image.fromarray(np.zeros([base_h, base_w], dtype=np.uint8))
            self.im_sgbm_trace_mean = Image.fromarray(np.zeros([base_h, base_w], dtype=np.uint8))
        # display the trace state
        if calc_delta:
            self.im_bm_trace_mean = Image.fromarray(np.array(np.minimum(self.bm_trace_sum, 255), np.uint8))
            self.im_sgbm_trace_mean = Image.fromarray(np.array(np.minimum(self.sgbm_trace_sum, 255), np.uint8))
        # update UI
        if self.master.attributes("-fullscreen"):
            self.im_disp_bm_r = self.im_bm_trace_mean.resize((588, 420))
        else:
            self.im_disp_bm_r = self.im_bm_trace_mean.resize((336, 240))
        self.tk_im_disp_bm = ImageTk.PhotoImage(self.im_disp_bm_r)
        self.canvas_disp_bm['image'] = self.tk_im_disp_bm
        if self.master.attributes("-fullscreen"):
            self.im_disp_sgbm_r = self.im_sgbm_trace_mean.resize((588, 420))
        else:
            self.im_disp_sgbm_r = self.im_sgbm_trace_mean.resize((336, 240))
        self.tk_im_disp_sgbm = ImageTk.PhotoImage(self.im_disp_sgbm_r)
        self.canvas_disp_sgbm['image'] = self.tk_im_disp_sgbm

    def trace_disp(self):
        if not self.trace_started:
            self.trace_started = True
            self.bar.set(0)
            self.bar.update()
            idx = int(self.bar_val.get())
            self.compare_disp(idx > 0)
        else:
            idx = int(self.bar_val.get())
            idx += 1
            if idx <= int(self.bar['to']):
                self.bar.set(idx)
                self.bar.update()
                self.compare_disp(idx > 0)
            else:
                self.trace_started = False

    def auto_trace_disp(self):
        self.trace_started = False
        for i in range(int(self.bar['to'])+1):
            self.trace_disp()


root = tk.Tk()
root.geometry('880x660+500+300')
root["background"] = "black"
#root.attributes("-fullscreen", True)
app = Application(master=root)
app.mainloop()