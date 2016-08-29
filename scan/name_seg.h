//
//  name_seg.h
//  plastic-CardIO
//
//  Created by Homer on 2016-08-29.
//
//

#ifndef name_seg_h
#define name_seg_h

#include "expiry_types.h"
#include "opencv2/imgproc/types_c.h"

void best_name_seg(IplImage *card_y, uint16_t starting_y_offset, GroupedRectsList &name_groups);


#endif /* name_seg_h */
