Extending
*********

Define A New Spectrograph
=========================

Each implemented spectrograph must have a file named spectrograph.py (must be lowercase) in the data module. This file can contain a variety of items, but it must contain the following methods and variables in order to support:

Reduction
+++++++++

.. code-block:: python


    categorize_raw_data(data_input_path: str, output_path: str) -> dict

    correct_readmath(data: Echellogram, data_image: np.ndarray)

    gen_master_calib_filename(master_cal: MasterCal)
    gen_master_calib_header(master_cal: MasterCal)

    group_darks(darks:list<MasterCal>)
    group_flats(flats:list<MasterCal>)

    pair_master_bias(data: RawEchellogram, master_bias: list<MasterCal>)
    pair_master_dark(data: RawEchellogram, master_darks: list<MasterCal>)
    pair_master_flat(data: RawEchellogram, master_flats: list<MasterCal>)
    pair_order_maps(data: RawEchellogram, order_maps: list<RawEchellogram>)

    parse_exposure_start_time(data: Echellogram)
    parse_fiber_nums(data: RawEchellogram)
    parse_image(data: Echellogram)
    parse_image_header(data: Echellogram)
    parse_image_num(data: RawEchellogram)
    parse_itime(data: Echellogram)
    parse_object(data: RawEchellogram)
    parse_sky_coord(data: Echellogram)
    parse_utdate(data: Echellogram)
    

Spectral Forward Modeling
+++++++++++++++++++++++++

.. code-block:: python

    compute_barycenter_corrections(data: Union[Spec1d, RawEchellogram], star_name: str)
    compute_exposure_midpoint(data: Union[Spec1d, RawEchellogram])
    estimate_wls(data: Union[Spec1d, RawEchellogram], order_num=None, fiber_num=None)
    parse_spec1d(data: Spec1d)