Extending
*********

Implementing a New Spectrograph
===============================

For each supported spectrograph, there is a unique ``spectrograph.py`` file within the module ``pychell.data``. This file contains all of the necessary details for each spectrograph. The great thing about Python is this file can be modified at runtime. In order to implement a spectrograph not officially supported, one has two options:

#. Create a new file ``pychell/data/spectrograph.py`` where spectrograph is the desired name of the spectrogaph (must be lowercase).
#. Use the ``generic`` spectrograph (especially useful for a one-off analysis) module by defining a few specific functions within ones own runtime.

It's also possible to start from an currently supported spectrograph and only modify the necessary methods or classes.

Below are default method names currently used in pychell.

Reduction
+++++++++

.. code-block:: python

    categorize_raw_data(data_input_path: str, output_path: str) -> dict

    correct_readmath(data: Echellogram, data_image: np.ndarray) -> np.ndarray

    gen_master_calib_filename(master_cal: MasterCal) -> str
    gen_master_calib_header(master_cal: MasterCal) -> Header

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


Below is an example of how one may implement or override: ``parse_itime``.

.. code-block:: python

    def parse_itime(data):
        return float(data.header["ITIME"])

    spectrograph.parse_itime = parse_itime