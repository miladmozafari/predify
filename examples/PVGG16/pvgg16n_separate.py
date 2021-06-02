from predify.modules import PCoderN
from predify.networks import PNetSeparateHP
from torch.nn import Sequential, ReLU, ConvTranspose2d

class PVGG16SeparateHP(PNetSeparateHP):
    def __init__(self, backbone, build_graph=False, random_init=False, ff_multiplier=(0.2,0.4,0.4,0.5,0.6), fb_multiplier=(0.05,0.1,0.1,0.1,0.0), er_multiplier=(0.01,0.01,0.01,0.01,0.01)):
        super().__init__(backbone, 5, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        # PCoder number 1
        pmodule = ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.pcoder1 = PCoderN(pmodule, True, self.random_init)
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.features[3].register_forward_hook(fw_hook1)

        # PCoder number 2
        pmodule = Sequential(ConvTranspose2d(128, 64, kernel_size=(10, 10), stride=(2, 2), padding=(4, 4)), ReLU(inplace=True))
        self.pcoder2 = PCoderN(pmodule, True, self.random_init)
        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.features[8].register_forward_hook(fw_hook2)

        # PCoder number 3
        pmodule = Sequential(ConvTranspose2d(256, 128, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder3 = PCoderN(pmodule, True, self.random_init)
        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.backbone.features[15].register_forward_hook(fw_hook3)

        # PCoder number 4
        pmodule = Sequential(ConvTranspose2d(512, 256, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder4 = PCoderN(pmodule, True, self.random_init)
        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm4, fbm=self.fbm4, erm=self.erm4)
            return e[0]
        self.backbone.features[22].register_forward_hook(fw_hook4)

        # PCoder number 5
        pmodule = Sequential(ConvTranspose2d(512, 512, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder5 = PCoderN(pmodule, False, self.random_init)
        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=None, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm5, fbm=self.fbm5, erm=self.erm5)
            return e[0]
        self.backbone.features[29].register_forward_hook(fw_hook5)


