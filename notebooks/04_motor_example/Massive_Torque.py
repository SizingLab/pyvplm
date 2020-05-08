# -*- coding: utf-8 -*-
"""
Created on Tue May 29 10:02:31 2018

@author: degiorgi
"""


def Couple_Mass(x, lb, ub, Kg, SDe, h, omega):
    import numpy as np
    from math import pi, cos, sin, asin, tan, sqrt, floor
    import femm

    SRi = x[0] * (ub[0] - lb[0]) + lb[0]  # Stator inner radius
    SLd = x[1] * (ub[1] - lb[1]) + lb[1]  # Tooth thickness
    SEp = x[2] * (ub[2] - lb[2]) + lb[2]  # Yoke thickness
    ALa = x[3] * (ub[3] - lb[3]) + lb[3]  # Magnets width
    RRi = x[4] * (ub[4] - lb[4]) + lb[4]  # Interior rotor radius
    J_den = x[5] * (ub[5] - lb[5]) + lb[5]  # Current density

    """ geometrical constraints NX310"""
    #    g1=SEp/SLd
    #    g2=SEp/SLd/0.3
    #    g3=(SRi+SEp)/26/Kg
    #    g4=SRi/SLd/7
    #    g5=SRi/SLd/2.6
    #    g6=SRi/SEp/13.6

    """ geometrical constraints NX110"""
    g1 = SEp / SLd
    g2 = SEp / SLd / 0.3
    g3 = (SRi + SEp) / 26 / Kg
    g4 = SRi / SLd / 7
    g5 = SRi / SLd / 2.6
    g6 = SRi / SEp / 13.6

    Kp = 10 ** 4  # Coefficient used to penalize the objective function

    if (g1 > 1) | (g2 < 1) | (g3 > 1) | (g4 > 1) | (g5 < 1) | (g6 > 1):
        f = Kp
    else:
        print("Stator De: {:.1f}mm".format(SDe))
        print("Stator Di: {:.1f}mm".format(SRi * 2))
        print("Stator e_tooth: {:.1f}mm".format(SLd))
        print("Stator e_yoke: {:.1f}mm".format(SEp))
        print("Magnets w_pm: {:.1f}mm".format(ALa))
        print("Rotor Ri: {:.1f}mm".format(RRi))

        """ Main motor parameter definition:"""

        SEt = 45  # Motor Lenght
        # SEt=87                             # Motor Lenght
        OffsetResolver = (
            -0.0
        )  # offset regulation of the resolver in mechanic angel for the rotor. Worst case: -1.5°

        Offset = OffsetResolver * pi / 180  # Offset angle in radiant

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Motor geometric variables definition (BLAC_parametres_geometrie)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        """Solver parameter"""
        AngleSommetMinMaillage = 10  # Angle au sommet min des éléments compris entre 1° et 33.8°
        Precision = 1e-008  # Précision relative comprise entre 1e-008 et 1e-016

        """Type of Command: (AC Flag_AC_DC=1) and (DC : Flag_AC_DC=0) """
        Flag_AC_DC = 1

        """Matirial: FlagBHpoint=0 calcul linéaire et FlagBHpoint=1 calcul non linéaire """
        FlagBHpoint = 1  # réutilisés dans Material_definition
        FlagBHpointAimant = 0
        FlagToleranceAimant = 2  # =1 mini		=2 nominal		=3 maxi

        """ Form factor definition: (relative to the exterior stator rayon)"""
        SRe = SDe / 2  # Exterior Stator rayon
        SRe_ref = 62 / 2  # Reference Exterior Stator rayon (From Parvex NX310)
        K = SRe / SRe_ref  # Form factor
        #        SRi=20.4*K             # Interior Stator rayon

        """ Airgap definition"""
        e = 0.4 * K  # Airgap thickness

        """Definition of the geometry which supports the magnets (Rotor)"""
        RRe_ref = (20.4 - 0.4) * K  # Exterior Rotor rayon of reference
        RRe = SRi - e  # Exterior Rotor rayon
        K2 = RRe / RRe_ref  # Rotor form factor (pour garder un entrefer constant)
        RAngDepSupport = 12 * pi / 180  # Angle de depouille du support à l'ouverture

        """Definition of the geometric parameters of the stator"""
        Neref = 12  # Reference Slot Number
        Ne = 12  # Slot Number
        KNe = (
            Ne / Neref
        )  # Coefficients ratio of the slots (if different slots are taked into account)
        NbDemiEncoche = 0  # Nomber of half-slots: 0 (Single Layer) ou 2 (Double Layer)
        ACwind = 0  # Sinisoidal winding: 0 (No sinusoidal winding) ou 1 (Sinusoidal winding)
        #        SLd=5.6*K                          # Tooth thickness
        #        SEp=3*K                           # Yoke thickness
        SRfe = SRe - SEp  # Stator rayon at the end of the slot
        SLa = 2.35 * K / KNe  # Opening slot width
        SLo = 0.6 * K / KNe  # Tooth thickness at the opening slot

        SAngDepEncoche = 15 * pi * KNe / 180  # Clearance angle of the tooth at the opening

        SHc = 1.00  # enfoncement au rayon des toles dans la culasse rapportee
        SHjx = 0.10  # jeu en x des toles dans la culasse rapportee
        SHjy = 0.10  # jeu en y des toles avec la culasse rapportee
        SENomex = 0.3 * K  # epaisseur du film de nomex faisant isolant dans l'encoche
        SRiNomex = 1.25 / KNe * K  # rayon intérieur du film de nomex
        SNShunt = 0 / 5  # dans le shunt : 1 tole sur 5
        SRatioLongueurActive = (
            0.97  # ratio donnant la longueur active du paquet de toles participant au couple
        )

        """ Definition of the geometry of the interior tube of rotor"""
        Npref = 10  # Reference Permanent magnet number
        Np = 10  # Permanent magnet number
        KNp = Np / Npref  # Coefficient ratio of the pole numbers
        #        RRi=8.75*K                          # Interior Rotor rayon

        """ Definition of the Permanent Magnets geometry"""
        RLo = (
            0.8 * K / KNp
        )  # longeur de la petite partie reliant tube interieur et support (jeu aimants-tube intérieur)
        RLa = 0.8 * K / KNp  # largeur de la petite partie reliant tube intérieur et support
        Npp = (
            1  # Nombre d'aimants unitaires pour constituer un pole (More then on if halbach motor)
        )
        #        ALa=3.3*K                           # Largeur aimants
        ALo = 0.88 * RRe - RRi  # Longeur aimants
        ARi = RRi + RLo  # Rayon intérieur des aimants
        ARe = ARi + ALo  # Rayon ext�rieur des aimants
        TempAimant = 20  # Température des aimants

        """ Definition of the Winding structure (A verifier:https://www.emetor.com/edit/windings) """
        BPeriodeBob = Ne
        if (Np == 10) | (Np == 4):
            if ACwind == 1:
                BNomBob = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
                BSigneBob = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
            elif NbDemiEncoche == 0:
                BNomBob = ["A", "A", "B", "B", "C", "C", "A", "A", "B", "B", "C", "C"]
                BSigneBob = np.array([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
            elif NbDemiEncoche == 2:
                BNomBob = [
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "B",
                    "B",
                    "C",
                    "C",
                    "C",
                    "C",
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "B",
                    "B",
                    "C",
                    "C",
                    "C",
                    "C",
                ]
                BSigneBob = np.array(
                    [
                        1,
                        -1,
                        -1,
                        1,
                        -1,
                        1,
                        1,
                        -1,
                        1,
                        -1,
                        -1,
                        1,
                        -1,
                        1,
                        1,
                        -1,
                        1,
                        -1,
                        -1,
                        1,
                        -1,
                        1,
                        1,
                        -1,
                    ]
                )
        if Np == 14:
            if ACwind == 1:
                BNomBob = ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
                BSigneBob = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
            elif NbDemiEncoche == 0:
                BNomBob = ["A", "A", "C", "C", "B", "B", "A", "A", "C", "C", "B", "B"]
                BSigneBob = np.array([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])
            elif NbDemiEncoche == 2:
                BNomBob = [
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "B",
                    "B",
                    "C",
                    "C",
                    "C",
                    "C",
                    "A",
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "B",
                    "B",
                    "C",
                    "C",
                    "C",
                    "C",
                ]
                BSigneBob = np.array(
                    [
                        1,
                        -1,
                        -1,
                        1,
                        -1,
                        1,
                        1,
                        -1,
                        1,
                        -1,
                        -1,
                        1,
                        -1,
                        1,
                        1,
                        -1,
                        1,
                        -1,
                        -1,
                        1,
                        -1,
                        1,
                        1,
                        -1,
                    ]
                )

        AngleInit = (2 * pi / Ne - 2 * pi / Np) / 2

        """ Mesh Definition:"""
        TailleMailleEntrefer = e / 2  # Cote utilisée pour définition matériau entrefer
        TailleMaille = 1.0 * K
        TailleMailleJeu = 0.25 * K
        TailleMailleBobine = 1 * K

        """ Definition of the exterior contour limiting the problem"""
        LRe = SRe * 1.5

        """ Groups Definition"""
        # 	aimants 					             : groupe 3
        # 	air à l'intérieur du rotor	         : groupe 1
        # 	culasse         				         : groupe 4
        # 	toles 						             : groupe 6
        #   dent                                 : groupe 5
        # 	bobines phase 1  plus			      : groupe 7
        # 	bobines phase 1 moins			      : groupe 7
        # 	bobines phase 2 plus			         : groupe 7
        # 	bobines phase 2 moins 			      : groupe 7
        # 	bobines phase 3 plus 			      : groupe 7
        # 	bobines phase 3 moins 			      : groupe 7
        # 	air dans les ouverture d'encoche	  : groupe 2
        # 	air dans l'entrefer 			         : groupe 2

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Calcul of the winding filling tax (Calcul_BOBINAGE)
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
        # caractéristiques bobinage selon diamétre (RESULTAT_optimisation_Dfil) => choisir le diamétre qui convient
        # calculs bobinage: resitance, longueur fil, puissance dissipée...

        """ Calcul des points définissant la géométrie de l'encoche """
        SReNomex = SRiNomex + SENomex  # Rayon de courbure extérieur du Nomex
        SAngElec = 2 * pi / Ne  # angle P O M

        """ Définition du raccordement du fond d'encoche avec la dent: points Eo E1 et E2"""
        SEx = SLd / 2  # point E (sans rayon de raccordement)
        SEy = (SRfe ** 2 - (SLd / 2) ** 2) ** 0.5  # point E (sans rayon de raccordement)
        SAngEOY = asin(SLd / 2 / SRfe)  # angle E Origine Axe Y
        SEox = SLd / 2 + SReNomex  # centre du cercle Eo
        SEoy = ((SRfe - SReNomex) ** 2 - SEox ** 2) ** 0.5  # centre du cercle Eo
        SAngEoOY = asin(SEox / (SRfe - SReNomex))  # angle Eo Origine Axe Y
        SAngL1pOEo = SAngElec - SAngEoOY * 2  # angle Lp Origine Eo
        SE1x = SRfe * sin(
            SAngEoOY
        )  # raccordement du fond d'encoche avec le rayon de raccordement E1
        SE1y = SRfe * cos(
            SAngEoOY
        )  # raccordement du fond d'encoche avec le rayon de raccordement E1
        SE2x = SLd / 2  # raccordement du rayon de raccordement evec la dent E2
        SE2y = SEoy  # raccordement du rayon de raccordement evec la dent E2
        SAngE1EoE2 = pi / 2 + SAngEoOY  # angle E1 Eo E2

        """ Définition du point H """
        SAngVOH = asin(SLa / 2 / SRi) * 2  # angle Ip O H
        SHx = SRi * (sin(SAngElec / 2 - SAngVOH / 2))  # point H
        SHy = SRi * (cos(SAngElec / 2 - SAngVOH / 2))  # point H
        SAngHOI = SAngElec - SAngVOH  # angle Ip Origine H

        """ Définition du point G """
        SGx = SHx + SLo * sin(SAngElec / 2 - SAngVOH / 2)  # point G
        SGy = SHy + SLo * cos(SAngElec / 2 - SAngVOH / 2)  # point G

        """ Définition du raccordement de la dent avec la forme polaire : points FO F1 et F2 """
        SAngYFG = pi / 2 + SAngDepEncoche + SAngElec / 2  # angle axe Y avec segment FG
        SFx = SLd / 2  # point F (sans rayon de raccordement)
        SFy = SGy + (SGx - SFx) * tan(
            SAngDepEncoche + SAngElec / 2
        )  # point F (sans rayon de raccordement)
        SFox = SLd / 2 + SReNomex  # centre du cercle FO
        SFoy = SFy + SReNomex / tan(SAngYFG / 2)  # centre du cercle FO
        SF1x = SFx  # raccordement de la dent avec la forme polaire : point F1
        SF1y = SFoy  # raccordement de la dent avec la forme polaire : point F1
        SF2x = SFox + SReNomex * cos(
            -(SAngDepEncoche + SAngElec / 2) - pi / 2
        )  # raccordement de la dent avec la forme polaire : point F2
        SF2y = SF1y + SReNomex * sin(
            -(SAngDepEncoche + SAngElec / 2) - pi / 2
        )  # raccordement de la dent avec la forme polaire : point F2

        SAngF1FoF2 = (pi / 2 - SAngYFG / 2) * 2  # angle F1 Fo F2
        """% fprintf(1,'SAngYFG = %f degr� \n',SAngYFG/pi*180);
        % fprintf(1,'SAngF1FoF2 = %f degr� \n',SAngF1FoF2/pi*180);"""

        """ Définition du point L miroir de E par rapport à Y """
        SLx = -SEx
        SLy = SEy

        """ Définition des points L1 et L2 miroir de E1 et E2 par rapport à Y """
        SL1x = -SE1x
        SL1y = SE1y
        SL2x = -SE2x
        SL2y = SE2y

        """ Définition du point I miroir de H par rapport à Y """
        SIx = -SHx
        SIy = SHy

        """ Définition du point J miroir de G par rapport à Y """
        SJx = -SGx
        SJy = SGy

        """ Définition des point K1 et K2 miroir de F1 et F2 par rapport à Y """
        SK1x = -SF1x
        SK1y = SF1y
        SK2x = -SF2x
        SK2y = SF2y

        """ Définition du point M """
        SMx = -SRfe * sin(SAngElec / 2)
        SMy = SRfe * cos(SAngElec / 2)

        """ Définition du point Z """
        SZx = -(SRe) * sin(SAngElec / 2)
        SZy = (SRe) * cos(SAngElec / 2)

        """ Définition du point W """
        SWx = -SZx
        SWy = SZy

        """ Définition du point N """
        SNx = -((SJx ** 2 + SJy ** 2) ** 0.5) * sin(SAngElec / 2)
        SNy = (SJx ** 2 + SJy ** 2) ** 0.5 * cos(SAngElec / 2)

        """ Définition du point P """
        SPx = -SMx
        SPy = SMy

        """ Définition du point U """
        SUx = -SNx
        SUy = SNy

        """ Définition des varaibles servant à selectionner les arcs et les segments """
        SPE1x = SRfe * sin((SAngElec / 2 + SAngEoOY) / 2)
        SPE1y = SRfe * cos((SAngElec / 2 + SAngEoOY) / 2)
        SE1E2x = SEox + SReNomex * cos(pi / 2 - SAngElec / 2 + SAngE1EoE2 / 2)
        SE1E2y = SEoy + SReNomex * sin(pi / 2 - SAngElec / 2 + SAngE1EoE2 / 2)
        SF1F2x = SFox + SReNomex * cos(pi + SAngF1FoF2 / 2)
        SF1F2y = SF1y + SReNomex * sin(pi + SAngF1FoF2 / 2)
        SL1Mx = -SPE1x
        SL1My = SPE1y
        SL1L2x = -SE1E2x
        SL1L2y = SE1E2y
        SK1K2x = -SF1F2x
        SK1K2y = SF1F2y
        SIJx = 0
        SIJy = SRi

        """ Définition des angles max pour les arcs """
        MaxSegDegOG = TailleMailleEntrefer / (SGx ** 2 + SGy ** 2) ** 0.5 * 360
        MaxSegDegE1E2 = 10
        MaxSegDegPE1 = SAngL1pOEo * 180 / pi / 10

        """ Définition de ces méme points pour la surface sans Nomex """
        """ points demi encoche de droite """
        SE1Ix = (SRfe - SENomex) * sin(SAngEoOY)  # Abscisse du point SE1I
        SE1Iy = (SRfe - SENomex) * cos(SAngEoOY)
        SE2Ix = SLd / 2 + SENomex
        SE2Iy = SEoy
        SF1Ix = SLd / 2 + SENomex
        SF1Iy = SF1y
        SF2Ix = SFox + (SReNomex - SENomex) * cos(-(SAngDepEncoche + SAngElec / 2) - pi / 2)
        SF2Iy = SF1y + (SReNomex - SENomex) * sin(-(SAngDepEncoche + SAngElec / 2) - pi / 2)
        SPIx = (SRfe - SENomex) * sin(SAngElec / 2)
        SPIy = (SRfe - SENomex) * cos(SAngElec / 2)
        SGIx = SHx + (SLo + SENomex) * sin(SAngElec / 2 - SAngVOH / 2)
        SGIy = SHy + (SLo + SENomex) * cos(SAngElec / 2 - SAngVOH / 2)
        SUIx = (SGIx ** 2 + SGIy ** 2) ** 0.5 * sin(SAngElec / 2)
        SUIy = (SGIx ** 2 + SGIy ** 2) ** 0.5 * cos(SAngElec / 2)

        """ variables pour arcs et segments de la demi encoche de droite """
        SPIE1Ix = (SRfe - SENomex) * sin((SAngElec / 2 + SAngEoOY) / 2)
        SPIE1Iy = (SRfe - SENomex) * cos((SAngElec / 2 + SAngEoOY) / 2)
        SE1IE2Ix = SEox + (SReNomex - SENomex) * cos(pi / 2 - SAngElec / 2 + SAngE1EoE2 / 2)
        SE1IE2Iy = SEoy + (SReNomex - SENomex) * sin(pi / 2 - SAngElec / 2 + SAngE1EoE2 / 2)
        SF1IF2Ix = SFox + (SReNomex - SENomex) * cos(pi + SAngF1FoF2 / 2)
        SF1IF2Iy = SFoy + (SReNomex - SENomex) * sin(pi + SAngF1FoF2 / 2)

        """ points demi bobine de gauche """
        SMIx = -SPIx
        SMIy = SPIy
        SL1Ix = -SE1Ix
        SL1Iy = SE1Iy
        SL2Ix = -SE2Ix
        SL2Iy = SE2Iy
        SK1Ix = -SF1Ix
        SK1Iy = SF1Iy
        SK2Ix = -SF2Ix
        SK2Iy = SF2Iy
        SJIx = -SGIx
        SJIy = SGIy
        SNIx = -SUIx
        SNIy = SUIy

        """ variables pour arcs et segments de la demi encoche de gauche """
        SMIL1Ix = -SPIE1Ix
        SMIL1Iy = SPIE1Iy
        SL1IL2Ix = -SE1IE2Ix
        SL1IL2Iy = SE1IE2Iy
        SK1IK2Ix = -SF1IF2Ix
        SK1IK2Iy = SF1IF2Iy

        """ CALCUL DE LA SURFACE ALLOUE AU BOBINAGE (Nomex compris : "temp_s2.fem" """
        femm.openfemm(1)
        femm.newdocument(0)  # probléme en magnétique
        femm.mi_probdef(0, "millimeters", "planar", Precision, SEt, AngleSommetMinMaillage)
        # Précision comprise entre 1e-008 et 1e-016
        # épaisseur 110mm à ajuster
        # Angle au sommet min des éléments compris entre 1° et 33.8°

        """ AIR """
        Mu_x = 1
        Mu_y = 1
        H_c = 0
        J = 0
        Cduct = 0
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "air", Mu_x, Mu_y, H_c, J, Cduct, Lam_d, Phi_max, Lam_fill, Lam_type, Phi_hx, Phi_hy
        )

        """ CONSTRUCTION DE LA SURFACE """
        femm.mi_addnode(SPx, SPy)
        femm.mi_addnode(SE1x, SE1y)
        femm.mi_addnode(SE2x, SE2y)
        femm.mi_addnode(SF1x, SF1y)
        femm.mi_addnode(SF2x, SF2y)
        femm.mi_addnode(SGx, SGy)
        femm.mi_addnode(SUx, SUy)
        #        femm.mi_addnode(SZx,SZy)
        #        femm.mi_addnode(SWx,SWy)

        femm.mi_addsegment(SE2x, SE2y, SF1x, SF1y)
        femm.mi_addsegment(SF2x, SF2y, SGx, SGy)
        femm.mi_addsegment(SGx, SGy, SUx, SUy)
        femm.mi_addsegment(SUx, SUy, SPx, SPy)
        femm.mi_addsegment(SMx, SMy, SZx, SZy)
        femm.mi_addsegment(SPx, SPy, SWx, SWy)

        femm.mi_addarc(SPx, SPy, SE1x, SE1y, SAngL1pOEo / 2 * 180 / pi, 1)
        femm.mi_addarc(SE1x, SE1y, SE2x, SE2y, SAngE1EoE2 * 180 / pi, 1)
        femm.mi_addarc(SF1x, SF1y, SF2x, SF2y, SAngF1FoF2 * 180 / pi, 1)

        femm.mi_selectnode(SPx, SPy)
        femm.mi_selectnode(SE1x, SE1y)
        femm.mi_selectnode(SE2x, SE2y)
        femm.mi_selectnode(SF1x, SF1y)
        femm.mi_selectnode(SF2x, SF2y)
        femm.mi_selectnode(SGx, SGy)
        femm.mi_selectnode(SUx, SUy)
        femm.mi_selectnode(SZx, SZy)
        femm.mi_selectnode(SWx, SWy)
        femm.mi_setnodeprop("TOTALE", 200)
        femm.mi_clearselected()

        femm.mi_selectsegment((SE2x + SF1x) / 2, (SE2y + SF1y) / 2)
        femm.mi_selectsegment((SF2x + SGx) / 2, (SF2y + SGy) / 2)
        femm.mi_selectsegment((SGx + SUx) / 2, (SGy + SUy) / 2)
        femm.mi_selectsegment((SUx + SPx) / 2, (SUy + SPy) / 2)
        femm.mi_selectsegment((SZx + SMx) / 2, (SZy + SMy) / 2)
        femm.mi_selectsegment((SWx + SPx) / 2, (SWy + SPy) / 2)
        femm.mi_setsegmentprop("TOTALE", TailleMailleBobine, 1, 0, 200)
        femm.mi_clearselected()

        femm.mi_selectarcsegment(SPE1x, SPE1y)
        femm.mi_selectarcsegment(SE1E2x, SE1E2y)
        femm.mi_selectarcsegment(SF1F2x, SF1F2y)
        femm.mi_setarcsegmentprop(MaxSegDegOG, "TOTALE", 0, 200)
        femm.mi_clearselected()

        femm.mi_addblocklabel((SPx + SF2x) / 2, (SPy + SF2y) / 2)
        femm.mi_selectlabel((SPx + SF2x) / 2, (SPy + SF2y) / 2)
        femm.mi_setblockprop("air", 0, TailleMailleEntrefer, 0, 0, 200, 1)
        femm.mi_clearselected()

        femm.mi_zoomnatural
        femm.mi_saveas("temp_s2.fem")
        femm.mi_analyze(0)
        femm.mi_loadsolution()
        femm.mo_smooth("on")
        femm.mo_groupselectblock(200)
        SEdemi_totale = femm.mo_blockintegral(5) * 1e6
        # SE_slot = SEdemi_totale*2
        SE_totale = SEdemi_totale * 2

        if NbDemiEncoche == 0:
            femm.mi_selectsegment((SUx + SPx) / 2, (SUy + SPy) / 2)
            femm.mi_deleteselectedsegments
            femm.mi_clearselected()

        femm.mi_purgemesh

        """ calcul du taux de remplissage de l'encoche """
        k_w = 0.24

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
                                    MODELISATION OF ROTOR """

        """ Problem definition """
        femm.newdocument(0)  # probléme en magnétique
        femm.mi_probdef(0, "millimeters", "planar", Precision, SEt, AngleSommetMinMaillage)

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        Material definition
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
        """ AIR """
        Mu_x = 1
        Mu_y = 1
        H_c = 0
        J = 0
        Cduct = 0
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "air", Mu_x, Mu_y, H_c, J, Cduct, Lam_d, Phi_max, Lam_fill, Lam_type, Phi_hx, Phi_hy
        )

        """ NOMEX """
        """ 	Mu_x = 1;
        % 	Mu_y = 1;
        % 	H_c = 0;
        % 	J = 0;
        % 	Cduct = 0; 
        % 	Lam_d = 0;
        % 	Phi_max = 0;  
        % 	Lam_fill = 1;
        % 	Lam_type = 0; 		% (0 :laminated in plane ; 3 : magnet wire)
        % 	Phi_hx = 0;
        % 	Phi_hy =0;
        % 	mi_addmaterial('Nomex',Mu_x,Mu_y ,H_c,J,Cduct,Lam_d,Phi_max,Lam_fill,Lam_type,Phi_hx,Phi_hy); """

        """ FIL DE CUIVRE """
        Mu_x = 1
        Mu_y = 1
        H_c = 0
        J = 0
        Cduct = 58  # (=58 : conductance du cuivre  ;  =0 : résistance du bobinage non pris en compte, à rajouter aprés)
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire) [AK]
        Phi_hx = 0
        Phi_hy = 0
        Nstrands = 0
        WireD = 1  # not considered in further calculation by FEMM [AK]

        femm.mi_addmaterial(
            "copper",
            Mu_x,
            Mu_y,
            H_c,
            J,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
            Nstrands,
            WireD,
        )

        """ specification for current density - J_den_tot """

        J_MatiereCuivre_Ap = sqrt(2) * J_den * k_w  # J_den_tot
        femm.mi_addmaterial(
            "A+",
            Mu_x,
            Mu_y,
            H_c,
            J_MatiereCuivre_Ap,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
            Nstrands,
            WireD,
        )
        J_MatiereCuivre_An = (-1) * sqrt(2) * J_den * k_w
        femm.mi_addmaterial(
            "A-",
            Mu_x,
            Mu_y,
            H_c,
            J_MatiereCuivre_An,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
            Nstrands,
            WireD,
        )

        J_MatiereCuivre_Bp = sqrt(2) * J_den * k_w * cos(2 * pi / 3)
        femm.mi_addmaterial(
            "B+",
            Mu_x,
            Mu_y,
            H_c,
            J_MatiereCuivre_Bp,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
            Nstrands,
            WireD,
        )
        J_MatiereCuivre_Bn = (-1) * sqrt(2) * J_den * k_w * cos(2 * pi / 3)
        femm.mi_addmaterial(
            "B-",
            Mu_x,
            Mu_y,
            H_c,
            J_MatiereCuivre_Bn,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
            Nstrands,
            WireD,
        )

        J_MatiereCuivre_Cp = sqrt(2) * J_den * k_w * cos(4 * pi / 3)
        femm.mi_addmaterial(
            "C+",
            Mu_x,
            Mu_y,
            H_c,
            J_MatiereCuivre_Cp,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
            Nstrands,
            WireD,
        )
        J_MatiereCuivre_Cn = (-1) * sqrt(2) * J_den * k_w * cos(4 * pi / 3)
        femm.mi_addmaterial(
            "C-",
            Mu_x,
            Mu_y,
            H_c,
            J_MatiereCuivre_Cn,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
            Nstrands,
            WireD,
        )

        """ MATIERE TOLES : MAAMAR """
        Mu_x = 1000
        Mu_y = 1000
        H_c = 0
        J = 0
        Cduct = 1 / 249.3e-9 / 1e6  # Conductivity (= 1/Resistivity) en Mega Siemens/metre
        Lam_d = 0.35
        Phi_max = 0
        Lam_fill = SRatioLongueurActive
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "FeSi M19 3%",
            Mu_x,
            Mu_y,
            H_c,
            J,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
        )
        BFeSi = np.array(
            [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
            ]
        )
        HFeSi = np.array(
            [
                0,
                9,
                17,
                26,
                35,
                43,
                52,
                60,
                71,
                88,
                111,
                145,
                213,
                446,
                1172,
                2865,
                5185,
                8405,
                13307,
                21050,
                54248,
                118110,
                197697,
                277265,
                356842,
                436420,
            ]
        )

        for ii in range(BFeSi.size):
            femm.mi_addbhpoint("FeSi M19 3%", BFeSi[ii], HFeSi[ii])

        """ MATIERE TOLES : ECEPS FeSi """
        Mu_x = 1000
        Mu_y = 1000
        H_c = 0
        J = 0
        Cduct = 1 / 249.3e-9 / 1e6  # Conductivity (= 1/Resistivity) en Mega Siemens/metre
        Lam_d = 0.35
        Phi_max = 0
        Lam_fill = SRatioLongueurActive
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "FeSi 0.35mm",
            Mu_x,
            Mu_y,
            H_c,
            J,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
        )
        Htole = np.array(
            [
                0,
                24,
                36,
                44,
                54,
                62,
                70,
                81,
                94,
                112,
                134,
                165,
                210,
                290,
                460,
                980,
                2660,
                5700,
                10700,
                18400,
                30000,
                45000,
                70000,
                120000,
                200000,
                280000,
            ]
        )
        Btole = np.array(
            [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
            ]
        )

        for i in range(Htole.size):
            femm.mi_addbhpoint("FeSi 0.35mm", Btole[i], Htole[i])

        """ MATIERE SUPPORT D'AIMANTS : MAAMAR """
        Mu_x = 2000
        Mu_y = 2000
        H_c = 0
        J = 0
        Cduct = 1.67  # (=1.67 : Losil 800/65)
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "Losil 800/65",
            Mu_x,
            Mu_y,
            H_c,
            J,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
        )
        BLosil = np.array(
            [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
            ]
        )
        HLosil = np.array(
            [
                0,
                18,
                37,
                55,
                74,
                84,
                96,
                110,
                130,
                155,
                190,
                240,
                320,
                450,
                680,
                1200,
                2200,
                5000,
                9000,
                15500,
                24000,
                36000,
                75789,
                155366,
                234944,
                314521,
            ]
        )
        for ii in range(BLosil.size):
            femm.mi_addbhpoint("Losil 800/65", BLosil[ii], HLosil[ii])

        """ MATIERE SUPPORT D'AIMANTS : ECEPS """
        Mu_x = 2000
        Mu_y = 2000
        H_c = 0
        J = 0
        Cduct = 1 / 130e-9 / 1e6  # Conductivity (= 1/Resistivity) en Mega Siemens/metre
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "Matiere_ROTOR",
            Mu_x,
            Mu_y,
            H_c,
            J,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
        )
        Hrotor = np.array(
            [
                0,
                42,
                70,
                79,
                86,
                95,
                106,
                118,
                132,
                149,
                176,
                210,
                274,
                347,
                576,
                791,
                2141,
                4187,
                8275,
                13500,
                23947,
                40320,
                120000,
                200000,
                280000,
                360000,
            ]
        )
        Brotor = np.array(
            [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
            ]
        )

        for ii in range(Hrotor.size):
            femm.mi_addbhpoint("Matiere_ROTOR", Brotor[ii], Hrotor[ii])

        """ MATIERE AIMANTS Sm2Co17 RECOMA 28 """
        Mu0 = 4 * pi / 1e7
        Mur = 1.05
        CoeffBrTemp = -0.035 / 100

        if FlagToleranceAimant == 1:
            Br20 = 1.04

        if FlagToleranceAimant == 2:
            Br20 = 1.07

        if FlagToleranceAimant == 3:
            Br20 = 1.10

            # 1.04 Tesla minimum  1.08 Tesla nominal for Parminder  1.07 Tesla nominal for Maamar

        Br = Br20 * (1 + CoeffBrTemp * (TempAimant - 20))
        Hcb = Br / Mur / Mu0
        BHmax = Br * Hcb / 4
        Cduct = 1 / ((0.75e6 + 0.9e6) / 2) / 1e6

        if FlagBHpointAimant == 0:
            Mu_x = Mur
            Mu_y = Mur
            H_c = Hcb
            J = 0
            Cduct = 1 / ((0.75e6 + 0.9e6) / 2) / 1e6
            Lam_d = 0
            Phi_max = 0
            Lam_fill = 1
            Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
            Phi_hx = 0
            Phi_hy = 0
            femm.mi_addmaterial(
                "Sm2Co17",
                Mu_x,
                Mu_y,
                H_c,
                J,
                Cduct,
                Lam_d,
                Phi_max,
                Lam_fill,
                Lam_type,
                Phi_hx,
                Phi_hy,
            )
        else:
            HAim = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            BAim = np.array([0, 0.1375, 0.275, 0.4125, 0.55, 0.6875, 0.825, 0.9625, 1.1])
            BSmCo = np.array([])
            HSmCo = np.array([])
            for ii in range(HAim.size):
                BSmCo.append(BAim[ii] + Br - BAim[9])
                HSmCo.append(HAim[ii] * 1e6)
                if BSmCo[1] < 0:
                    BSmCo[1] = 0
            Mu_x = Mur
            Mu_y = Mur
            H_c = HSmCo[9]
            J = 0
            Cduct = 1 / ((0.75e6 + 0.9e6) / 2) / 1e6
            Lam_d = 0
            Phi_max = 0
            Lam_fill = 1
            Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
            Phi_hx = 0
            Phi_hy = 0
            femm.mi_addmaterial(
                "Sm2Co17 Recoma 28",
                Mu_x,
                Mu_y,
                H_c,
                J,
                Cduct,
                Lam_d,
                Phi_max,
                Lam_fill,
                Lam_type,
                Phi_hx,
                Phi_hy,
            )
            for ii in range(BSmCo.size):
                femm.mi_addbhpoint("Sm2Co17 Recoma 28", BSmCo[ii], HSmCo[ii])

        """ MATIERE TOLES """
        Mu_x = 1000
        Mu_y = 1000
        H_c = 0
        J = 0
        Cduct = 1 / 249.3e-9 / 1e6  # Conductivity (= 1/Resistivity) en Mega Siemens/metre
        Lam_d = 0.35
        Phi_max = 0
        Lam_fill = SNShunt * SRatioLongueurActive
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "FeSi M19 3% SHUNT",
            Mu_x,
            Mu_y,
            H_c,
            J,
            Cduct,
            Lam_d,
            Phi_max,
            Lam_fill,
            Lam_type,
            Phi_hx,
            Phi_hy,
        )
        BFeSi = np.array(
            [
                0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
            ]
        )
        HFeSi = np.array(
            [
                0,
                9,
                17,
                26,
                35,
                43,
                52,
                60,
                71,
                88,
                111,
                145,
                213,
                446,
                1172,
                2865,
                5185,
                8405,
                13307,
                21050,
                54248,
                118110,
                197697,
                277265,
                356842,
                436420,
            ]
        )
        for ii in range(BFeSi.size):
            femm.mi_addbhpoint("FeSi M19 3% SHUNT", BFeSi[ii], HFeSi[ii])

        """ MATIERE TUBE INTERIEUR AU ROTOR : CX13VDW mesuré par le CEDRAT """
        Mu_x = 2000
        Mu_y = 2000
        H_c = 0
        J = 0
        Cduct = 1 / 77e-8 / 1e6  # Conductivity (= 1/Resistivity) en Mega Siemens/metre
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "CX13VDW", Mu_x, Mu_y, H_c, J, Cduct, Lam_d, Phi_max, Lam_fill, Lam_type, Phi_hx, Phi_hy
        )
        HCX13VDW = np.array(
            [
                0,
                262,
                720,
                1049,
                1532,
                1863,
                2315,
                2799,
                3092,
                3614,
                3914,
                4388,
                4888,
                5216,
                5689,
                5995,
                6770,
                13561,
                20121,
                27517,
                37246,
                48142,
                86000,
                166000,
                246000,
                326000,
                406000,
                486000,
                566000,
                646000,
                726000,
                806000,
            ]
        )
        BCX13VDW = np.array(
            [
                0,
                0.0118,
                0.0295,
                0.0407,
                0.0634,
                0.0793,
                0.1148,
                0.1639,
                0.2069,
                0.3221,
                0.4083,
                0.5155,
                0.5981,
                0.6449,
                0.7016,
                0.7341,
                0.8026,
                1.1120,
                1.2626,
                1.3768,
                1.4829,
                1.5527,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
            ]
        )
        for ii in range(BCX13VDW.size):
            femm.mi_addbhpoint("CX13VDW", BCX13VDW[ii], HCX13VDW[ii])

        """" MATIERE TUBE INTERIEUR AU ROTOR : XD15NW mesuré par le CEDRAT """
        Mu_x = 2000
        Mu_y = 2000
        H_c = 0
        J = 0
        Cduct = 1 / 77e-8 / 1e6  # Conductivity (= 1/Resistivity) en Mega Siemens/metre
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  #         # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "XD15NW", Mu_x, Mu_y, H_c, J, Cduct, Lam_d, Phi_max, Lam_fill, Lam_type, Phi_hx, Phi_hy
        )
        HXD15NW = np.array(
            [
                0,
                359,
                902,
                1230,
                1738,
                2043,
                2520,
                2898,
                3351,
                3848,
                4174,
                4701,
                4953,
                5512,
                5795,
                6333,
                6897,
                13965,
                20704,
                29830,
                37490,
                43295,
                61200,
                221200,
                381200,
                541200,
                701200,
                861200,
                1021200,
            ]
        )
        BXD15NW = np.array(
            [
                0,
                0.0134,
                0.0242,
                0.0335,
                0.0467,
                0.0577,
                0.0740,
                0.0849,
                0.1064,
                0.1337,
                0.1652,
                0.2252,
                0.2893,
                0.3918,
                0.4529,
                0.5216,
                0.5938,
                0.8822,
                0.9902,
                1.0813,
                1.1379,
                1.1776,
                1.2,
                1.4,
                1.6,
                1.8,
                2.0,
                2.2,
                2.4,
            ]
        )
        for ii in range(BXD15NW.size):
            femm.mi_addbhpoint("XD15NW", BXD15NW[ii], HXD15NW[ii])
        """ MATIERE TUBE INTERIEUR AU ROTOR : BS-S98 donnée par ECEPS """
        Mu_x = 2000
        Mu_y = 2000
        H_c = 0
        J = 0
        Cduct = 1 / 77e-8 / 1e6  # Conductivity (= 1/Resistivity) en Mega Siemens/metre
        Lam_d = 0
        Phi_max = 0
        Lam_fill = 1
        Lam_type = 0  # (0 :laminated in plane ; 3 : magnet wire)
        Phi_hx = 0
        Phi_hy = 0
        femm.mi_addmaterial(
            "BS-S98", Mu_x, Mu_y, H_c, J, Cduct, Lam_d, Phi_max, Lam_fill, Lam_type, Phi_hx, Phi_hy
        )
        HBSS98 = np.array(
            [
                0,
                80,
                239,
                398,
                557,
                796,
                955,
                1194,
                1592,
                3183,
                3979,
                4775,
                7958,
                11937,
                15915,
                19894,
                99894,
                179894,
                259894,
                339894,
                419894,
                499894,
            ]
        )
        BBSS98 = np.array(
            [
                0,
                0.013,
                0.03,
                0.05,
                0.08,
                0.17,
                0.35,
                0.8,
                1.185,
                1.59,
                1.66,
                1.72,
                1.805,
                1.835,
                1.84,
                1.845,
                1.945,
                2.045,
                2.145,
                2.245,
                2.345,
                2.445,
            ]
        )
        for ii in range(BBSS98.size):
            femm.mi_addbhpoint("BS-S98", BBSS98[ii], HBSS98[ii])

        """ ANNULER LES POINT B-H POUR TRAVAILLER EN LINEAIRE """
        if FlagBHpoint == 0:
            femm.mi_clearbhpoints("FeSi M19 3%")
            femm.mi_clearbhpoints("FeSi 0.35mm")
            femm.mi_clearbhpoints("Losil 800/65")
            femm.mi_clearbhpoints("Matiere_ROTOR")
            femm.mi_clearbhpoints("Sm2Co17")
            femm.mi_clearbhpoints("FeSi M19 3% SHUNT")
            femm.mi_clearbhpoints("CX13VDW")
            femm.mi_clearbhpoints("XD15NW")
            femm.mi_clearbhpoints("BS-S98")

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                         trace_ROTOR
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        """ CALCUL DES COORDONNEES DES POINTS SERVANT A LA CONSTRUCTION DU ROTOR """

        """ calcul des points définissant la géométrie du support """
        RAngElec = 2 * pi / Np  # angle P O M

        """ Définition de la partie extérieur du support: point E """
        REx = ALa / 2  # point E
        REy = (RRe ** 2 - REx ** 2) ** 0.5  # point E
        RE1x = REx - 0.5 * K
        RE1y = REy - 0.1 * K
        RE2x = RE1x + 0.05 * K
        RE2y = RE1y - 0.7 * K
        RE3x = REx - 0.2 * K
        RE3y = RE2y
        RE4x = REx
        RE4y = RE2y - 0.2 * K

        """ Définition des angles """
        RAngEOY = asin(REx / RRe)  # angle E Origine Axe Y
        RAngLOE = RAngElec - RAngEOY * 2  # angle L Origine E

        """ Définition du point H """
        RAngVOH = asin(RLa / 2 / RRi) * 2  # angle Ip O H
        RAngHOI = RAngElec - RAngVOH  # angle Ip Origine H
        RH1x = (RRi + 0.6 * K) * (sin(RAngElec / 2 - RAngVOH / 2))  # point H1
        RH1y = (RRi + 0.6 * K) * (cos(RAngElec / 2 - RAngVOH / 2))
        RH2x = ALa / 2
        RH2y = RRi

        """ Définition du point G """
        RGx = RH1x + RLo * sin(RAngElec / 2)
        # point G
        RGy = RH1y + RLo * cos(RAngElec / 2)
        # point G

        """ Définition du point F """
        RFx = ALa / 2  # point F (sans rayon de raccordement)
        RFy = ARi

        """ Définition du point L miroir de E par rapport et Y """
        RLx = -REx
        RLy = REy
        RL1x = -RE1x
        RL1y = RE1y
        RL2x = -RE2x
        RL2y = RE2y
        RL3x = -RE3x
        RL3y = RE3y
        RL4x = -RE4x
        RL4y = RE4y

        " Définition du point I miroir de H par rapport à Y " ""
        RI1x = -RH1x
        RI1y = RH1y
        RI2x = -RH2x
        RI2y = RH2y

        """ Définition du point J miroir de G par rapport à Y """
        RJx = -RGx
        RJy = RGy

        """ Définition des point K1 et K2 miroir de F1 et F2 par rapport à Y """
        RKx = -RFx
        RKy = RFy

        """ Définition du point M	"""
        RMx = -RRe * sin(RAngElec / 2)
        RMy = RRe * cos(RAngElec / 2)

        """ Définition du point P """
        RPx = -RMx
        RPy = RMy

        """ Définition des variables servant à la selection des arcs et des segments """
        RPEx = RRe * sin((RAngElec / 2 + RAngEOY) / 2)
        RPEy = RRe * cos((RAngElec / 2 + RAngEOY) / 2)
        RLMx = -RPEx
        RLMy = RPEy
        RHIx = 0
        RHIy = RRi

        """ Définition des points de la partie entre le tube intérieur et l'aimant  """
        R1x = 0.4 * K
        R1y = RRi
        R2x = R1x
        R2y = ARi
        R3x = -R2x
        R3y = R2y
        R4x = -R1x
        R4y = R1y

        """ partie superieure amant """
        A1x = ALa / 2
        A1y = ARe - 0.3 * K
        A4x = -A1x
        A4y = A1y

        """ trace support aimant """
        RAnglep = 0 - 2 * pi / Np
        RPxrot = RMx * cos(RAnglep) - RMy * sin(RAnglep)
        RPyrot = RMx * sin(RAnglep) + RMy * cos(RAnglep)
        femm.mi_addnode(RPxrot, RPyrot)
        femm.mi_selectnode(RPxrot, RPyrot)
        femm.mi_setnodeprop("rotor", 9)
        femm.mi_clearselected()

        for AngleDeg in range(0, 360 + Np, int(360 / Np)):
            Angle = AngleDeg * pi / 180
            S = sin(Angle)
            C = cos(Angle)
            RExrot = REx * C - REy * S
            REyrot = REx * S + REy * C
            RE1xrot = RE1x * C - RE1y * S
            RE1yrot = RE1x * S + RE1y * C
            RE2xrot = RE2x * C - RE2y * S
            RE2yrot = RE2x * S + RE2y * C
            RE3xrot = RE3x * C - RE3y * S
            RE3yrot = RE3x * S + RE3y * C
            RE4xrot = RE4x * C - RE4y * S
            RE4yrot = RE4x * S + RE4y * C
            RFxrot = RFx * C - RFy * S
            RFyrot = RFx * S + RFy * C
            RH1xrot = RH1x * C - RH1y * S
            RH1yrot = RH1x * S + RH1y * C
            RH2xrot = RH2x * C - RH2y * S
            RH2yrot = RH2x * S + RH2y * C
            RI1xrot = RI1x * C - RI1y * S
            RI1yrot = RI1x * S + RI1y * C
            RI2xrot = RI2x * C - RI2y * S
            RI2yrot = RI2x * S + RI2y * C
            RGxrot = RGx * C - RGy * S
            RGyrot = RGx * S + RGy * C
            RJxrot = RJx * C - RJy * S
            RJyrot = RJx * S + RJy * C
            RKxrot = RKx * C - RKy * S
            RKyrot = RKx * S + RKy * C
            RLxrot = RLx * C - RLy * S
            RLyrot = RLx * S + RLy * C
            RL1xrot = RL1x * C - RL1y * S
            RL1yrot = RL1x * S + RL1y * C
            RL2xrot = RL2x * C - RL2y * S
            RL2yrot = RL2x * S + RL2y * C
            RL3xrot = RL3x * C - RL3y * S
            RL3yrot = RL3x * S + RL3y * C
            RL4xrot = RL4x * C - RL4y * S
            RL4yrot = RL4x * S + RL4y * C
            RMxrot = RMx * C - RMy * S
            RMyrot = RMx * S + RMy * C
            RPExrot = RPEx * C - RPEy * S
            RPEyrot = RPEx * S + RPEy * C
            RLMxrot = RLMx * C - RLMy * S
            RLMyrot = RLMx * S + RLMy * C
            RIJxrot = RHIx * C - RHIy * S
            RIJyrot = RHIx * S + RHIy * C
            R1xrot = R1x * C - R1y * S
            R1yrot = R1x * S + R1y * C
            R2xrot = R2x * C - R2y * S
            R2yrot = R2x * S + R2y * C
            R3xrot = R3x * C - R3y * S
            R3yrot = R3x * S + R3y * C
            R4xrot = R4x * C - R4y * S
            R4yrot = R4x * S + R4y * C

            A1xrot = A1x * C - A1y * S
            A1yrot = A1x * S + A1y * C
            A4xrot = A4x * C - A4y * S
            A4yrot = A4x * S + A4y * C

            femm.mi_addnode(RExrot, REyrot)
            femm.mi_addnode(RE1xrot, RE1yrot)
            femm.mi_addnode(RE2xrot, RE2yrot)
            femm.mi_addnode(RE3xrot, RE3yrot)
            femm.mi_addnode(RE4xrot, RE4yrot)
            femm.mi_addnode(RLxrot, RLyrot)
            femm.mi_addnode(RL1xrot, RL1yrot)
            femm.mi_addnode(RL2xrot, RL2yrot)
            femm.mi_addnode(RL3xrot, RL3yrot)
            femm.mi_addnode(RL4xrot, RL4yrot)
            femm.mi_addnode(RMxrot, RMyrot)
            femm.mi_addnode(A1xrot, A1yrot)
            femm.mi_addnode(A4xrot, A4yrot)

            femm.mi_addsegment(RE1xrot, RE1yrot, RExrot, REyrot)
            femm.mi_addsegment(RE1xrot, RE1yrot, RE2xrot, RE2yrot)
            femm.mi_addsegment(RE2xrot, RE2yrot, RE3xrot, RE3yrot)
            femm.mi_addsegment(RE3xrot, RE3yrot, RE4xrot, RE4yrot)
            femm.mi_addsegment(RL1xrot, RL1yrot, RLxrot, RLyrot)
            femm.mi_addsegment(RL1xrot, RL1yrot, RL2xrot, RL2yrot)
            femm.mi_addsegment(RL2xrot, RL2yrot, RL3xrot, RL3yrot)
            femm.mi_addsegment(RL3xrot, RL3yrot, RL4xrot, RL4yrot)
            femm.mi_addsegment(RL4xrot, RL4yrot, A4xrot, A4yrot)
            femm.mi_addsegment(RE4xrot, RE4yrot, A1xrot, A1yrot)

            femm.mi_addarc(RPxrot, RPyrot, RExrot, REyrot, RAngLOE / 2 * 180 / pi, 1)
            femm.mi_addarc(RLxrot, RLyrot, RMxrot, RMyrot, RAngLOE / 2 * 180 / pi, 1)

            femm.mi_selectnode(RExrot, REyrot)
            femm.mi_selectnode(RE1xrot, RE1yrot)
            femm.mi_selectnode(RE2xrot, RE2yrot)
            femm.mi_selectnode(RE3xrot, RE3yrot)
            femm.mi_selectnode(RE4xrot, RE4yrot)
            femm.mi_selectnode(RKxrot, RKyrot)
            femm.mi_selectnode(RLxrot, RLyrot)
            femm.mi_selectnode(RL1xrot, RL1yrot)
            femm.mi_selectnode(RL2xrot, RL2yrot)
            femm.mi_selectnode(RL3xrot, RL3yrot)
            femm.mi_selectnode(RL4xrot, RL4yrot)
            femm.mi_selectnode(RMxrot, RMyrot)
            femm.mi_setnodeprop("rotor", 9)
            femm.mi_clearselected()

            femm.mi_selectsegment((RE1xrot + RExrot) / 2, (RE1yrot + REyrot) / 2)
            femm.mi_selectsegment((RE1xrot + RE2xrot) / 2, (RE1yrot + RE2yrot) / 2)
            femm.mi_selectsegment((RE2xrot + RE3xrot) / 2, (RE2yrot + RE3yrot) / 2)
            femm.mi_selectsegment((RE4xrot + RE3xrot) / 2, (RE4yrot + RE3yrot) / 2)
            femm.mi_selectsegment((RFxrot + RGxrot) / 2, (RFyrot + RGyrot) / 2)
            femm.mi_selectsegment((RH1xrot + RGxrot) / 2, (RH1yrot + RGyrot) / 2)
            femm.mi_selectsegment((RH2xrot + R1xrot) / 2, (RH2yrot + R1yrot) / 2)
            femm.mi_selectsegment((R2xrot + R1xrot) / 2, (R2yrot + R1yrot) / 2)
            femm.mi_selectsegment((R2xrot + R3xrot) / 2, (R2yrot + R3yrot) / 2)
            femm.mi_selectsegment((R3xrot + R4xrot) / 2, (R3yrot + R4yrot) / 2)
            femm.mi_selectsegment((R4xrot + RI2xrot) / 2, (R4yrot + RI2yrot) / 2)
            femm.mi_selectsegment((RI1xrot + RJxrot) / 2, (RI1yrot + RJyrot) / 2)
            femm.mi_selectsegment((RJxrot + RKxrot) / 2, (RJyrot + RKyrot) / 2)
            femm.mi_selectsegment((RL1xrot + RLxrot) / 2, (RL1yrot + RLyrot) / 2)
            femm.mi_selectsegment((RL1xrot + RL2xrot) / 2, (RL1yrot + RL2yrot) / 2)
            femm.mi_selectsegment((RL2xrot + RL3xrot) / 2, (RL2yrot + RL3yrot) / 2)
            femm.mi_selectsegment((RL4xrot + RL3xrot) / 2, (RL4yrot + RL3yrot) / 2)
            femm.mi_selectsegment((RKxrot + RL4xrot) / 2, (RKyrot + RL4yrot) / 2)
            femm.mi_setsegmentprop("rotor", TailleMaille, 1, 0, 9)
            femm.mi_clearselected()

            femm.mi_selectarcsegment(RPExrot, RPEyrot)
            femm.mi_selectarcsegment(RLMxrot, RLMyrot)
            femm.mi_setarcsegmentprop(MaxSegDegPE1, "rotor", 0, 9)
            femm.mi_clearselected()

            RPxrot = RMxrot
            RPyrot = RMyrot
            A = (ALa / 2 + 2 * pi * (ARi + ALo / 2) / Np) / 2
            B = ARi + ALo / 2
            Ax = A * C - S * B
            Ay = A * S + B * C
            femm.mi_addblocklabel(Ax, Ay)
            femm.mi_selectlabel(Ax, Ay)
            MatiereSupportAimant = "FeSi 0.35mm"
            femm.mi_setblockprop(MatiereSupportAimant, 0, TailleMaille, 0, 0, 9, 1)
            femm.mi_clearselected()

        """ Définition points pour construction aimants """
        A1x = ALa / 2
        A1y = ARe - 0.3 * K
        A2x = A1x - 0.3 * K
        A2y = ARe
        A3x = -A2x
        A3y = A2y
        A4x = -A1x
        A4y = A1y
        A8x = A1x
        A8y = ARi + 0.3 * K
        A7x = A2x
        A7y = ARi
        A5x = -A8x
        A5y = A8y
        A6x = -A7x
        A6y = A7y

        AngleA1OA2 = 45

        """ trace AIMANTS """

        Sens = pi
        for AngleDeg in range(0, 360 + Np, int(360 / Np)):
            Angle = AngleDeg * pi / 180
            S = sin(Angle)
            C = cos(Angle)

            A1xrot = A1x * C - A1y * S
            A1yrot = A1x * S + A1y * C
            A2xrot = A2x * C - A2y * S
            A2yrot = A2x * S + A2y * C
            A3xrot = A3x * C - A3y * S
            A3yrot = A3x * S + A3y * C
            A4xrot = A4x * C - A4y * S
            A4yrot = A4x * S + A4y * C
            A5xrot = A5x * C - A5y * S
            A5yrot = A5x * S + A5y * C
            A6xrot = A6x * C - A6y * S
            A6yrot = A6x * S + A6y * C
            A7xrot = A7x * C - A7y * S
            A7yrot = A7x * S + A7y * C
            A8xrot = A8x * C - A8y * S
            A8yrot = A8x * S + A8y * C

            femm.mi_addnode(A1xrot, A1yrot)
            femm.mi_addnode(A2xrot, A2yrot)
            femm.mi_addnode(A3xrot, A3yrot)
            femm.mi_addnode(A4xrot, A4yrot)
            femm.mi_addnode(A5xrot, A5yrot)
            femm.mi_addnode(A6xrot, A6yrot)
            femm.mi_addnode(A7xrot, A7yrot)
            femm.mi_addnode(A8xrot, A8yrot)

            femm.mi_addsegment(A2xrot, A2yrot, A3xrot, A3yrot)
            femm.mi_addsegment(A4xrot, A4yrot, A5xrot, A5yrot)
            femm.mi_addsegment(A6xrot, A6yrot, A7xrot, A7yrot)
            femm.mi_addsegment(A8xrot, A8yrot, A1xrot, A1yrot)

            if AngleDeg >= 360 / Np:
                femm.mi_addsegment(A7xrot, A7yrot, A9x, A9y)

            femm.mi_addarc(A1xrot, A1yrot, A2xrot, A2yrot, AngleA1OA2, 1)
            femm.mi_addarc(A3xrot, A3yrot, A4xrot, A4yrot, AngleA1OA2, 1)
            femm.mi_addarc(A5xrot, A5yrot, A6xrot, A6yrot, AngleA1OA2, 1)
            femm.mi_addarc(A7xrot, A7yrot, A8xrot, A8yrot, AngleA1OA2, 1)

            femm.mi_selectnode(A1xrot, A1yrot)
            femm.mi_selectnode(A2xrot, A2yrot)
            femm.mi_selectnode(A3xrot, A3yrot)
            femm.mi_selectnode(A4xrot, A4yrot)
            femm.mi_selectnode(A5xrot, A5yrot)
            femm.mi_selectnode(A6xrot, A6yrot)
            femm.mi_selectnode(A7xrot, A7yrot)
            femm.mi_selectnode(A8xrot, A8yrot)
            femm.mi_setnodeprop("aimants", 3)
            femm.mi_clearselected()

            femm.mi_selectsegment((A2xrot + A3xrot) / 2, (A2yrot + A3yrot) / 2)
            femm.mi_selectsegment((A4xrot + A5xrot) / 2, (A4yrot + A5yrot) / 2)
            femm.mi_selectsegment((A6xrot + A7xrot) / 2, (A6yrot + A7yrot) / 2)
            femm.mi_selectsegment((A8xrot + A1xrot) / 2, (A8yrot + A1yrot) / 2)

            if AngleDeg >= 360 / Np:
                femm.mi_selectsegment((A7xrot + A9x) / 2, (A7yrot + A9y) / 2)

            femm.mi_setsegmentprop("aimant", TailleMaille, 1, 0, 3)
            femm.mi_clearselected()

            femm.mi_selectarcsegment(A2xrot, A2yrot)
            femm.mi_selectarcsegment(A3xrot, A3yrot)
            femm.mi_selectarcsegment(A5xrot, A5yrot)
            femm.mi_selectarcsegment(A7xrot, A7yrot)
            MaxSegDeg = 2 * asin(TailleMailleEntrefer / 2 / RRe) * 180 / pi
            femm.mi_setarcsegmentprop(MaxSegDeg, "aimant", 0, 3)
            femm.mi_clearselected()
            # segmento sopra i magneti
            femm.mi_selectsegment((A4xrot + RL4xrot) / 2, (A4yrot + RL4yrot) / 2)
            femm.mi_selectsegment((A1xrot + RE4xrot) / 2, (A1yrot + RE4yrot) / 2)
            femm.mi_setsegmentprop("rotor", TailleMaille, 1, 0, 3)
            femm.mi_clearselected()

            femm.mi_addblocklabel((A3xrot + A7xrot) / 2, (A3yrot + A7yrot) / 2)
            femm.mi_selectlabel((A3xrot + A7xrot) / 2, (A3yrot + A7yrot) / 2)
            MatiereAimant = "Sm2Co17"
            femm.mi_setblockprop(
                MatiereAimant, 0, TailleMaille, 0, AngleDeg + Sens * 180 / pi, 3, 1
            )
            femm.mi_clearselected()

            Sens = Sens + pi

            A9x = A6xrot
            A9y = A6yrot
            femm.mi_addnode(A9x, A9y)

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
                                        MODELISATION OF STATOR """

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                          trace_PAQUET_TOLES
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        """ trace cercle de rayon Sre """
        femm.mi_addnode(SRe, 0)
        femm.mi_addnode(-SRe, 0)
        femm.mi_addarc(-SRe, 0, SRe, 0, 180, 1)
        femm.mi_addarc(SRe, 0, -SRe, 0, 180, 1)

        femm.mi_addnode(SZx, SZy)
        femm.mi_addnode(SWx, SWy)

        femm.mi_selectnode(SRe, 0)
        femm.mi_selectnode(-SRe, 0)
        femm.mi_setnodeprop("TOLE", 2)
        femm.mi_clearselected()

        femm.mi_selectarcsegment(0, SRe)
        femm.mi_selectarcsegment(0, -SRe)
        MaxSegDeg = 2 * asin(TailleMaille / 2 / SRe) * 180 / pi
        femm.mi_setarcsegmentprop(MaxSegDeg, "TOLE", 0, 2)
        femm.mi_clearselected()

        """ trace encoches """
        Anglep = 0 - 2 * pi / Ne
        SPxrot = SMx * cos(Anglep) - SMy * sin(Anglep)
        SPyrot = SMx * sin(Anglep) + SMy * cos(Anglep)
        femm.mi_addnode(SPxrot, SPyrot)
        femm.mi_selectnode(SPxrot, SPyrot)
        femm.mi_setnodeprop("TOLE", 2)
        femm.mi_clearselected()

        for AngleDeg in range(0, 360, int(360 / Ne)):
            Angle = AngleDeg * pi / 180
            S = sin(Angle)
            C = cos(Angle)
            SE1xrot = SE1x * C - SE1y * S
            SE1yrot = SE1x * S + SE1y * C
            SE2xrot = SE2x * C - SE2y * S
            SE2yrot = SE2x * S + SE2y * C
            SF1xrot = SF1x * C - SF1y * S
            SF1yrot = SF1x * S + SF1y * C
            SF2xrot = SF2x * C - SF2y * S
            SF2yrot = SF2x * S + SF2y * C
            SGxrot = SGx * C - SGy * S
            SGyrot = SGx * S + SGy * C
            SHxrot = SHx * C - SHy * S
            SHyrot = SHx * S + SHy * C
            SIxrot = SIx * C - SIy * S
            SIyrot = SIx * S + SIy * C
            SJxrot = SJx * C - SJy * S
            SJyrot = SJx * S + SJy * C
            SK1xrot = SK1x * C - SK1y * S
            SK1yrot = SK1x * S + SK1y * C
            SK2xrot = SK2x * C - SK2y * S
            SK2yrot = SK2x * S + SK2y * C
            SL1xrot = SL1x * C - SL1y * S
            SL1yrot = SL1x * S + SL1y * C
            SL2xrot = SL2x * C - SL2y * S
            SL2yrot = SL2x * S + SL2y * C
            SMxrot = SMx * C - SMy * S
            SMyrot = SMx * S + SMy * C
            SNxrot = SNx * C - SNy * S
            SNyrot = SNx * S + SNy * C
            SUxrot = SUx * C - SUy * S
            SUyrot = SUx * S + SUy * C
            SPE1xrot = SPE1x * C - SPE1y * S
            SPE1yrot = SPE1x * S + SPE1y * C
            SL1Mxrot = SL1Mx * C - SL1My * S
            SL1Myrot = SL1Mx * S + SL1My * C
            SIJxrot = SIJx * C - SIJy * S
            SIJyrot = SIJx * S + SIJy * C
            SE1E2xrot = SE1E2x * C - SE1E2y * S
            SE1E2yrot = SE1E2x * S + SE1E2y * C
            SF1F2xrot = SF1F2x * C - SF1F2y * S
            SF1F2yrot = SF1F2x * S + SF1F2y * C
            SL1L2xrot = SL1L2x * C - SL1L2y * S
            SL1L2yrot = SL1L2x * S + SL1L2y * C
            SK1K2xrot = SK1K2x * C - SK1K2y * S
            SK1K2yrot = SK1K2x * S + SK1K2y * C
            femm.mi_addnode(SE1xrot, SE1yrot)
            femm.mi_addnode(SE2xrot, SE2yrot)
            femm.mi_addnode(SF1xrot, SF1yrot)
            femm.mi_addnode(SF2xrot, SF2yrot)
            femm.mi_addnode(SGxrot, SGyrot)
            femm.mi_addnode(SHxrot, SHyrot)
            femm.mi_addnode(SIxrot, SIyrot)
            femm.mi_addnode(SJxrot, SJyrot)
            femm.mi_addnode(SK1xrot, SK1yrot)
            femm.mi_addnode(SK2xrot, SK2yrot)
            femm.mi_addnode(SL1xrot, SL1yrot)
            femm.mi_addnode(SL2xrot, SL2yrot)
            femm.mi_addnode(SMxrot, SMyrot)
            femm.mi_addnode(SNxrot, SNyrot)
            femm.mi_addnode(SUxrot, SUyrot)

            femm.mi_addsegment(SE2xrot, SE2yrot, SF1xrot, SF1yrot)
            femm.mi_addsegment(SF2xrot, SF2yrot, SGxrot, SGyrot)
            femm.mi_addsegment(SGxrot, SGyrot, SHxrot, SHyrot)
            femm.mi_addsegment(SIxrot, SIyrot, SJxrot, SJyrot)
            femm.mi_addsegment(SJxrot, SJyrot, SK2xrot, SK2yrot)
            femm.mi_addsegment(SK1xrot, SK1yrot, SL2xrot, SL2yrot)

            femm.mi_addarc(SPxrot, SPyrot, SE1xrot, SE1yrot, SAngL1pOEo / 2 * 180 / pi, 1)
            femm.mi_addarc(SE1xrot, SE1yrot, SE2xrot, SE2yrot, SAngE1EoE2 * 180 / pi, 1)
            femm.mi_addarc(SF1xrot, SF1yrot, SF2xrot, SF2yrot, SAngF1FoF2 * 180 / pi, 1)
            femm.mi_addarc(SHxrot, SHyrot, SIxrot, SIyrot, SAngHOI * 180 / pi, 1)
            femm.mi_addarc(SL1xrot, SL1yrot, SMxrot, SMyrot, SAngL1pOEo / 2 * 180 / pi, 1)
            femm.mi_addarc(SL2xrot, SL2yrot, SL1xrot, SL1yrot, SAngE1EoE2 * 180 / pi, 1)
            femm.mi_addarc(SK2xrot, SK2yrot, SK1xrot, SK1yrot, SAngF1FoF2 * 180 / pi, 1)
            femm.mi_addarc(SJxrot, SJyrot, SNxrot, SNyrot, SAngL1pOEo / 2 * 180 / pi, 1)
            femm.mi_addarc(SGxrot, SGyrot, SUxrot, SUyrot, SAngL1pOEo / 2 * 180 / pi, 1)

            femm.mi_selectnode(SE1xrot, SE1yrot)
            femm.mi_selectnode(SE2xrot, SE2yrot)
            femm.mi_selectnode(SF1xrot, SF1yrot)
            femm.mi_selectnode(SF2xrot, SF2yrot)
            femm.mi_selectnode(SGxrot, SGyrot)
            femm.mi_selectnode(SHxrot, SHyrot)
            femm.mi_selectnode(SIxrot, SIyrot)
            femm.mi_selectnode(SJxrot, SJyrot)
            femm.mi_selectnode(SK1xrot, SK1yrot)
            femm.mi_selectnode(SK2xrot, SK2yrot)
            femm.mi_selectnode(SL1xrot, SL1yrot)
            femm.mi_selectnode(SL2xrot, SL2yrot)
            femm.mi_selectnode(SMxrot, SMyrot)
            femm.mi_selectnode(SNxrot, SNyrot)
            femm.mi_selectnode(SUxrot, SUyrot)
            femm.mi_setnodeprop("TOLE", 6)
            femm.mi_clearselected()

            femm.mi_selectsegment((SE2xrot + SF1xrot) / 2, (SE2yrot + SF1yrot) / 2)
            femm.mi_selectsegment((SF2xrot + SGxrot) / 2, (SF2yrot + SGyrot) / 2)
            femm.mi_selectsegment((SJxrot + SK2xrot) / 2, (SJyrot + SK2yrot) / 2)
            femm.mi_selectsegment((SK1xrot + SL2xrot) / 2, (SK1yrot + SL2yrot) / 2)
            femm.mi_setsegmentprop("TOLE", TailleMaille, 1, 0, 6)
            femm.mi_clearselected()

            femm.mi_selectsegment((SGxrot + SHxrot) / 2, (SGyrot + SHyrot) / 2)
            femm.mi_selectsegment((SIxrot + SJxrot) / 2, (SIyrot + SJyrot) / 2)
            femm.mi_setsegmentprop("TOLE", TailleMailleEntrefer, 1, 0, 6)
            femm.mi_clearselected()

            femm.mi_selectarcsegment(SPE1xrot, SPE1yrot)
            femm.mi_selectarcsegment(SL1Mxrot, SL1Myrot)
            femm.mi_setarcsegmentprop(MaxSegDegPE1, "TOLE", 0, 6)
            femm.mi_clearselected()

            femm.mi_selectarcsegment(SE1E2xrot, SE1E2yrot)
            femm.mi_selectarcsegment(SF1F2xrot, SF1F2yrot)
            femm.mi_selectarcsegment(SL1L2xrot, SL1L2yrot)
            femm.mi_selectarcsegment(SK1K2xrot, SK1K2yrot)
            femm.mi_setarcsegmentprop(MaxSegDegE1E2, "TOLE", 0, 6)
            femm.mi_clearselected()

            femm.mi_selectarcsegment(SIJxrot, SIJyrot)
            MaxSegDeg = 2 * asin(TailleMailleEntrefer / 2 / SRi) * 180 / pi
            femm.mi_setarcsegmentprop(MaxSegDeg, "TOLE", 0, 6)
            femm.mi_clearselected()

            SPxrot = SMxrot
            SPyrot = SMyrot

        femm.mi_addsegment(SMx, SMy, SZx, SZy)
        femm.mi_addsegment(SPx, SPy, SWx, SWy)
        femm.mi_addsegment(SL2x, SL2y, SE2x, SE2y)

        femm.mi_addblocklabel(0, (SRe + SRfe) / 2)
        femm.mi_selectlabel(0, (SRe + SRfe) / 2)
        MatiereTole = "FeSi 0.35mm"
        femm.mi_setblockprop(MatiereTole, 0, TailleMaille, 0, 0, 4, 1)
        femm.mi_clearselected()

        femm.mi_addblocklabel(0, SRi + (SRfe - SRi) / 4)
        femm.mi_selectlabel(0, SRi + (SRfe - SRi) / 4)
        MatiereTole = "FeSi 0.35mm"
        femm.mi_setblockprop(MatiereTole, 0, TailleMaille, 0, 0, 5, 1)
        femm.mi_clearselected()

        femm.mi_addblocklabel((SRe + SRfe) / 2, 0)
        femm.mi_selectlabel((SRe + SRfe) / 2, 0)
        femm.MatiereTole = "FeSi 0.35mm"
        femm.mi_setblockprop(MatiereTole, 0, TailleMaille, 0, 0, 6, 1)
        femm.mi_clearselected()

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                          trace_ENTREFER
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        """ AIR DANS L'ENTREFER : construction par nappe """

        IndEntreferMax = floor((SRi - ARe) / TailleMailleEntrefer)
        IndEntreferMax = 3
        for IndREntrefer in range(1, IndEntreferMax + 1):
            REntrefer = RRe + (SRi - RRe) / IndEntreferMax * IndREntrefer
            MaxSegDeg = TailleMailleEntrefer / REntrefer * 360
            if REntrefer == SRi:
                femm.mi_addblocklabel(
                    0, REntrefer - TailleMailleEntrefer / 2
                )  # tailleMailleEntrefer � changer si probl�me de d�finition mat�riau
                femm.mi_selectlabel(0, REntrefer - TailleMailleEntrefer / 2)
                femm.mi_setblockprop("air", 0, e / 4, 0, 0, 2, 1)
                femm.mi_clearselected()
            else:
                femm.mi_addnode(REntrefer, 0)
                femm.mi_addnode(-REntrefer, 0)
                femm.mi_addarc(REntrefer, 0, -REntrefer, 0, 180, 1)
                femm.mi_addarc(-REntrefer, 0, REntrefer, 0, 180, 1)

                femm.mi_selectnode(REntrefer, 0)
                femm.mi_selectnode(-REntrefer, 0)
                femm.mi_setnodeprop("ENTREFER", 2)
                femm.mi_clearselected()

                femm.mi_selectarcsegment(0, -REntrefer)
                femm.mi_selectarcsegment(0, REntrefer)
                femm.mi_setarcsegmentprop(MaxSegDeg, "ENTREFER", 0, 2)
                femm.mi_clearselected()
                femm.mi_addblocklabel(0, REntrefer - TailleMailleEntrefer / 2)
                femm.mi_selectlabel(0, REntrefer - TailleMailleEntrefer / 2)
                if IndREntrefer == 1:
                    NumeroGroupe = 1
                else:
                    NumeroGroupe = 2

                femm.mi_setblockprop("air", 0, TailleMailleEntrefer, 0, 0, NumeroGroupe, 1)
                femm.mi_clearselected()
        femm.mi_clearselected()

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                         trace_AIR_INTERIEUR_ROTOR
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
        MateriauTubeInterieur = "air"
        femm.mi_clearselected()
        femm.mi_addblocklabel(0, 0)
        femm.mi_selectlabel(0, 0)
        femm.mi_setblockprop("air", 0, 2, 0, 0, 1, 1)
        femm.mi_clearselected()

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                          trace_BOBINE_simple
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        """ BOBINE PHASE 1 """

        Anglep = 0 - 2 * pi / Ne
        SUxrot = SNx * cos(Anglep) - SNy * sin(Anglep)
        SUyrot = SNx * sin(Anglep) + SNy * cos(Anglep)
        SPxrot = SMx * cos(Anglep) - SMy * sin(Anglep)
        SPyrot = SMx * sin(Anglep) + SMy * cos(Anglep)

        if NbDemiEncoche == 0:
            for jjNe in range(Ne):
                Angle = 2 * pi / Ne * (jjNe)
                S = sin(Angle)
                C = cos(Angle)
                SPxrot = SPx * C - SPy * S
                SPyrot = SPx * S + SPy * C
                SUxrot = SUx * C - SUy * S
                SUyrot = SUx * S + SUy * C
                femm.mi_addblocklabel((SUxrot + SPxrot) / 2, (SUyrot + SPyrot) / 2)
                femm.mi_selectlabel((SUxrot + SPxrot) / 2, (SUyrot + SPyrot) / 2)
                #
                Sens = BSigneBob[jjNe]
                Phase = BNomBob[jjNe]

                if Phase == "A":
                    if Sens == 1:
                        femm.mi_setblockprop("A+", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("A-", 0, TailleMaille, 0, 0, 7, 1)
                if Phase == "B":
                    if Sens == 1:
                        femm.mi_setblockprop("B+", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("B-", 0, TailleMaille, 0, 0, 7, 1)
                if Phase == "C":
                    if Sens == 1:
                        femm.mi_setblockprop("C+", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("C-", 0, TailleMaille, 0, 0, 7, 1)
                femm.mi_clearselected()
        """         ------------------------------------------------------- """
        if NbDemiEncoche == 2:
            jjBper = 1
            for jjNe in range(Ne):
                Angle = 2 * pi / Ne * (jjNe)
                jjPer = mod(jjNe - 1, BPeriodeBob) + 1
                jjBob1 = 2 * (jjPer - 1) + 1
                jjBob2 = 2 * (jjPer - 1) + 2

                S = sin(Angle)
                C = cos(Angle)

                SPxrot = SPx * C - SPy * S
                SPyrot = SPx * S + SPy * C

                SUxrot = SUx * C - SUy * S
                SUyrot = SUx * S + SUy * C

                SF2xrot = SF2x * C - SF2y * S
                SF2yrot = SF2x * S + SF2y * C

                SK2xrot = SK2x * C - SK2y * S
                SK2yrot = SK2x * S + SK2y * C

                SMxrot = SMx * C - SMy * S
                SMyrot = SMx * S + SMy * C

                SNxrot = SNx * C - SNy * S
                SNyrot = SNx * S + SNy * C

                #       Demi bobine de droite par rapport à la dent

                femm.mi_addsegment(SUxrot, SUyrot, SPxrot, SPyrot)
                femm.mi_selectsegment((SUxrot + SPxrot) / 2, (SUyrot + SPyrot) / 2)
                femm.mi_setsegmentprop("BOBINE", TailleMailleBobine, 1, 0, 7)
                femm.mi_clearselected()

                femm.mi_addblocklabel((SPxrot + SF2xrot) / 2, (SPyrot + SF2yrot) / 2)
                femm.mi_selectlabel((SPxrot + SF2xrot) / 2, (SPyrot + SF2yrot) / 2)

                Sens = BSigneBob[jjBper]
                Phase = BNomBob[jjBper]

                if Phase == "A":
                    if Sens == 1:
                        femm.mi_setblockprop("A+", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("A-", 0, TailleMaille, 0, 0, 7, 1)
                    if Phase == "B":
                        if Sens == 1:
                            femm.mi_setblockprop("B+", 0, TailleMaille, 0, 0, 7, 1)
                        else:
                            femm.mi_setblockprop("B-", 0, TailleMaille, 0, 0, 7, 1)
                    if Phase == "C":
                        if Sens == 1:
                            femm.mi_setblockprop("C+", 0, TailleMaille, 0, 0, 7, 1)
                        else:
                            femm.mi_setblockprop("C-", 0, TailleMaille, 0, 0, 7, 1)

                #       Demi bobine de gauche par rapport à la dent """
                femm.mi_addsegment(SMxrot, SMyrot, SNxrot, SNyrot)
                femm.mi_selectsegment((SMxrot + SNxrot) / 2, (SMyrot + SNyrot) / 2)
                femm.mi_setsegmentprop("BOBINE", TailleMailleBobine, 1, 0, 7)
                femm.mi_clearselected()

                femm.mi_addblocklabel((SMxrot + SK2xrot) / 2, (SMyrot + SK2yrot) / 2)
                femm.mi_selectlabel((SMxrot + SK2xrot) / 2, (SMyrot + SK2yrot) / 2)
                Sens = BSigneBob(1, jjBper + 1)
                Phase = BNomBob(1, jjBper + 1)

                if Phase == "A":
                    if Sens == 1:
                        femm.mi_setblockprop("A+", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("A-", 0, TailleMaille, 0, 0, 7, 1)
                if Phase == "B":
                    if Sens == 1:
                        femm.mi_setblockprop("B+", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("B-", 0, TailleMaille, 0, 0, 7, 1)
                if Phase == "C":
                    if Sens == 1:
                        femm.mi_setblockprop("C+", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("C-", 0, TailleMaille, 0, 0, 7, 1)

                femm.mi_addblocklabel((SMxrot + SNxrot) / 2, (SMyrot + SNyrot) / 2)
                femm.mi_selectlabel((SMxrot + SNxrot) / 2, (SMyrot + SNyrot) / 2)

                Sens = BSigneBob[jjBper + 1]
                Phase = BNomBob[jjBper + 1]
                if Phase == "A":
                    if Sens == 1:
                        femm.mi_setblockprop("MatiereCuivre_pA", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("MatiereCuivre_nA", 0, TailleMaille, 0, 0, 7, 1)
                if Phase == "B":
                    if Sens == 1:
                        femm.mi_setblockprop("MatiereCuivre_pB", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("MatiereCuivre_nB", 0, TailleMaille, 0, 0, 7, 1)
                if Phase == "C":
                    if Sens == 1:
                        femm.mi_setblockprop("MatiereCuivre_pC", 0, TailleMaille, 0, 0, 7, 1)
                    else:
                        femm.mi_setblockprop("MatiereCuivre_nC", 0, TailleMaille, 0, 0, 7, 1)

                jjBper = jjBper + 2
        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        femm.mi_selectgroup(3)
        femm.mi_selectgroup(9)
        femm.mi_moverotate(0, 0, 0)

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
             MODELISATION LIMITE PROBLEME (air autour du moteur) trace_LIMITE_PROBLEME
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        femm.mi_addboundprop("Mixed", 0, 0, 0, 0, 0, 0, 1 / (4 * pi * 1e-7) / 100, 0, 8)
        femm.mi_addnode(LRe, 0)
        femm.mi_addnode(-LRe, 0)
        femm.mi_addarc(LRe, 0, -LRe, 0, 180, 2)
        femm.mi_addarc(-LRe, 0, LRe, 0, 180, 2)
        femm.mi_addblocklabel((LRe + SRe) / 2, 0)
        femm.mi_selectlabel((LRe + SRe) / 2, 0)
        femm.mi_setblockprop("air", 0, 2, 0, 0, 8, 1)
        femm.mi_clearselected()
        femm.mi_selectarcsegment(0, LRe)
        femm.mi_selectarcsegment(0, -LRe)
        femm.mi_setarcsegmentprop(1, "Mixed", 1, 8)
        femm.mi_clearselected()

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        Zoom + sauvegarde + Calcul Couple
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        femm.mi_zoomnatural()
        femm.mi_zoom(-SRe, -SRe, SRe, SRe)
        femm.mi_saveas("Moteur_BLAC.fem")
        femm.mi_seteditmode("group")
        femm.mi_createmesh()
        femm.mi_zoom(-1.05 * SRe, -1.05 * SRe, 1.05 * SRe, 1.05 * SRe)
        femm.mi_saveas("Moteur_BLAC.fem")
        femm.mi_analyze(0)
        femm.mi_loadsolution()
        femm.mo_smooth("on")
        femm.mo_groupselectblock(1)
        femm.mo_groupselectblock(3)
        femm.mo_groupselectblock(9)
        torque22 = femm.mo_blockintegral(22)
        MatCouple = torque22
        print("Electro-magnetic linear Torque: {:.2E}N".format(MatCouple / SEt))

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                            Calcul_PerteFer
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
        femm.mo_clearblock()
        femm.mo_clearcontour()

        """ Calcul des pertes Fer dans le moteur """

        """ Calcul des aires de mesure de B """

        femm.mo_groupselectblock(5)
        S1 = femm.mo_blockintegral(5)  # surface dent
        femm.mo_clearblock()

        femm.mo_groupselectblock(4)
        S2 = femm.mo_blockintegral(5)  # surface culasse
        femm.mo_clearblock()

        P1x = 0
        P1y = SRi + (SRfe - SRi) / 2
        P2x = -(SRfe + SEp / 2) * sin(SAngElec / 2)
        P2y = (SRfe + SEp / 2) * cos(SAngElec / 2)
        A = 0
        massevolumiquefer = 7650
        B_max = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )  # Calcul B max entre B_culasse et B_dent; Largeur vector=Ne;
        u = 0
        """ Mesure de B dans chaque aire tout autour du moteur """
        for AngleDeg in range(0, 360, int(360 / Ne)):
            Angle = AngleDeg * pi / 180
            S = sin(Angle)
            C = cos(Angle)

            P1xrot = P1x * C - P1y * S
            P1yrot = P1x * S + P1y * C
            P2xrot = P2x * C - P2y * S
            P2yrot = P2x * S + P2y * C

            femm.mi_addnode(P1xrot, P1yrot)
            femm.mi_addnode(P2xrot, P2yrot)

            femm.mi_selectnode(P1xrot, P1yrot)
            valeur1 = femm.mo_getb(P1xrot, P1yrot)
            B1 = (valeur1[0] ** 2 + valeur1[1] ** 2) ** 0.5

            femm.mi_selectnode(P2xrot, P2yrot)
            valeur2 = femm.mo_getb(P2xrot, P2yrot)
            B2 = (valeur2[0] ** 2 + valeur2[1] ** 2) ** 0.5
            Bm = max(B1, B2)
            B_max[u] = Bm
            u = u + 1
            if u < Ne + 2:  #% (u < Ne+2) pour Ne different de 12
                k_adt = 1  # coeff perte fer dent
                k_ady = 1
                # coeff perte fer culasse
                A = A + (SEt * 10 ** -3) * massevolumiquefer * (
                    k_adt * B1 ** 2 * S1 + k_ady * B2 ** 2 * S2
                )

        maxB = np.max(B_max)
        m_ref_1 = (SEt * 10 ** -3) * massevolumiquefer * S1
        m_ref_2 = (SEt * 10 ** -3) * massevolumiquefer * S2

        """ ---------------Joule losses------------------ """

        BResistivite = 1.7241 * 1e-8  # [Ohm*m]
        BS_cu = SE_totale * k_w  # surface Copper
        #  SE_encoche(Nind,1) = SE_totale     # surface encoche
        SE_encoche = SE_totale

        S = sin(2 * pi / Ne)
        C = cos(2 * pi / Ne)

        if ACwind == 1:
            P = sqrt(SPx ** 2 + SPy ** 2)
            U = sqrt(SUx ** 2 + SUy ** 2)
            SRslot_m = (P + U) / 2
            L_end = 2 * pi * SRslot_m / Np  # Longeur tract inactive
            BVol_end = 2 * BS_cu * L_end  # Volume bobinage inactive
            BVol_slot = SEt * BS_cu  # Volume bobinage active
            BVol_tot = BVol_slot + BVol_end  # Volum de Bobinage [mm^3]
            # P_Joule(1,Nind)  = 3*Ne/3*BResistivite*(BVol_tot*1e-9)*(J_den*(1/1e-6))^2
            P_Joule = 3 * Ne / 3 * BResistivite * (BVol_tot * 1e-9) * (J_den * (1 / 1e-6)) ** 2
        elif NbDemiEncoche == 0:
            SPxrot = SPx * C - SPy * S
            SPyrot = SPx * S + SPy * C
            SUxrot = SUx * C - SUy * S
            SUyrot = SUx * S + SUy * C
            SRslot_m = sqrt(SUxrot ** 2 + SUyrot ** 2)
            BR_chignon = SRslot_m * sin(pi / Ne)  # R_Toroide
            # l(Nind,1)=(SRslot_m*sin(pi/Ne))*10^-3
            Br_chignon = sqrt(BS_cu / pi)  # r-Toroide
            BVol_end = 2 * pi ** 2 * BR_chignon * Br_chignon ** 2  # Volume bobinage inactive
            BVol_slot = 2 * SEt * BS_cu  # Volume bobinage active
            BVol_tot = BVol_slot + BVol_end  # Volum de Bobinage [mm^3]
            # P_Joule(1,Nind) = 3/2*Ne/3*BResistivite*(BVol_tot*1e-9)*(J_den*(1/1e-6))^2
            P_Joule = 3 / 2 * Ne / 3 * BResistivite * (BVol_tot * 1e-9) * (J_den * (1 / 1e-6)) ** 2
        elif NbDemiEncoche == 2:
            SRslot_m = sqrt(SK2Ix ** 2 + SK2Iy ** 2)
            BR_chignon = SRslot_m * sin(pi / Ne)  # R_Toroide
            Br_chignon = sqrt(BS_cu / pi)  # r-Toroide
            BVol_end = 2 * pi ** 2 * BR_chignon * Br_chignon ** 2  # Volume bobinage inactive
            BVol_slot = 2 * SEt * BS_cu  # Volume bobinage active
            BVol_tot = BVol_slot + BVol_end  # Volum of Bobinage [mm^3]
            # P_Joule(1,Nind)  = 3/2*Ne/3*BResistivite*(BVol_tot*1e-9)*(J_den*(1/1e-6))^2
            P_Joule = 3 / 2 * Ne / 3 * BResistivite * (BVol_tot * 1e-9) * (J_den * (1 / 1e-6)) ** 2
        print("Joule Losses: {:.2E}W/m".format(P_Joule / SEt))

        """ ------------------------------------------- """

        """  Calcul fréquence rotation """
        freq_rot = omega / 60  # en Hz
        freq_rot_rad = freq_rot * 2 * pi  # en rad/s
        freq_mag = freq_rot * Np / 2  # freq_mag=freq_rot*nb_paires_poles

        P_fer_50_1 = 1.25  # Pertes fer normalisées à 50Hz et 1 Tesla
        # P_fer_femm(1,Nind)= P_fer_50_1*(freq_mag/50)^1.5*A % Pertes Fer calcul�es   --- iron losses
        P_fer_femm = (
            P_fer_50_1 * (freq_mag / 50) ** 1.5 * A
        )  # Pertes Fer calculées   --- iron losses
        print("Iron Losses: {:.2E}W/m".format(P_fer_femm / SEt))

        """ real torque [AK] """
        # MatCouple_real(1,Nind)=MatCouple(1,Nind) - P_fer_femm(1,Nind)./freq_rot_rad;
        MatCouple_real = MatCouple - P_fer_femm / freq_rot_rad

        """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                            Calcul_masse
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """

        femm.mo_clearblock()

        """ masse cuivre """
        mv_Cu = 8400  # masse volumique cuivre [Kg/m3]
        femm.mo_groupselectblock(7)
        VCu_active = femm.mo_blockintegral(10) * k_w  # volume partie active
        VCu_end = BVol_end * 1e-9
        # volume toroide
        if ACwind == 1:
            VCu = VCu_active + Ne * VCu_end
        else:
            VCu = VCu_active + Ne / 2 * VCu_end

        # MCu(1,Nind)= VCu*mv_Cu                      # masse cuivre
        MCu = VCu * mv_Cu  # masse cuivre
        femm.mo_clearblock()

        """ masse ferre """
        mv_Fe = 7650
        # masse volumique fer [Kg/m3]
        femm.mo_groupselectblock(9)
        femm.mo_groupselectblock(6)
        femm.mo_groupselectblock(4)
        femm.mo_groupselectblock(5)
        VFe = femm.mo_blockintegral(10)  # volume fer
        # MFe(1,Nind)= VFe*mv_Fe      # masse fer
        MFe = VFe * mv_Fe  # masse fer
        femm.mo_clearblock()

        """ masse aimants """
        mv_Sm = 8300  # masse volumique aimants [Kg/m3]
        femm.mo_groupselectblock(3)
        VSm = femm.mo_blockintegral(10)  # voulume aimants
        # MSm(1,Nind)= VSm*mv_Sm                     # masse aimants
        MSm = VSm * mv_Sm  # masse aimants
        femm.mo_clearblock()

        """ masse resine """
        mv_Re = 1200  # masse volumique resine [Kg/m3]
        VRe = VCu * (1 - k_w)  # volume resine
        MRe = VRe * mv_Re  # masse resine

        # Mtot(1,Nind)= MCu(1,Nind)+MFe(1,Nind)+MSm(1,Nind)+MRe(1,Nind)  % Mass Moteur
        Mtot = MCu + MFe + MSm + MRe
        print("Motor linear mass: {:.2f}kg/m".format(Mtot / SEt))

        """ Calculation of the objective function to minimize"""
        C1 = 63.7  # Constraint to satisfate: Pj < 63.71
        C2 = 4 / 7 * P_Joule  # Constraint to satisfate: Pfe < 4/7*Pj
        C3 = 1.8  # Constraint to satisfate: maxB < 1.8
        P1 = max(0, (P_Joule - C1) / C1)  # Penalized objective function
        P2 = max(0, (P_fer_femm - C2) / C2)  # Penalized objective function
        P3 = max(0, (maxB - C3) / C3)  # Penalized objective function
        P123 = np.array([P1, P2, P3])
        P = np.sum([P123])
        f = Mtot / torque22 + Kp * P  # Objective Function
        print("Penalized torque density: {:.2E}N.m/kg".format(1 / f))
        print("Penality: {:.2f}".format(P))
        print("\n")
    return f
