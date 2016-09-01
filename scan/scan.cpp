//
//  scan.cpp
//  See the file "LICENSE.md" for the full license governing this code.
//

#include "compile.h"
#if COMPILE_DMZ

#include "scan.h"
#include "expiry_categorize.h"
#include "expiry_seg.h"

#define SCAN_FOREVER 0  // useful for performance profiling
#define EXTRA_TIME_FOR_EXPIRY_IN_MICROSECONDS 1000 // once the card number has been successfully identified, allow a bit more time to figure out the expiry

#define kDecayFactor 0.8f
#define kMinStability 0.7f

static int counterForFailToGetName = 0.0;
static char imageStringForName[64];

void scanner_initialize(ScannerState *state) {
  scanner_reset(state);
}

void scanner_reset(ScannerState *state) {
  state->count15 = 0;
  state->count16 = 0;
  state->aggregated15 = NumberScores::Zero();
  state->aggregated16 = NumberScores::Zero();
  scan_analytics_init(&state->session_analytics);
  state->timeOfCardNumberCompletionInMilliseconds = 0;
  state->scan_expiry = false;
  state->expiry_month = 0;
  state->expiry_year = 0;
  state->expiry_groups.clear();
  state->name_groups.clear();
}

void scanner_add_frame(ScannerState *state, IplImage *y, FrameScanResult *result) {
  scanner_add_frame_with_expiry(state, y, false, result);
}

void getPredicationOfNameGroup(IplImage *card_y,GroupedRectsList &name_groups,std::vector<TensorFlowPredication> &preds){
  if (name_groups.empty()) {
    return;
  }
  
  std::vector<TensorFlowPredication> temp;
  static IplImage *as_float = NULL;
  if (as_float == NULL) {
    as_float = cvCreateImage(cvSize(kNameCharacterImageWidth, kNameCharacterImageHeight), IPL_DEPTH_8U, 1);
  }
  
  float newSum = 0.0;
  bool newHasUnder = false;
  GroupedRects groupR = name_groups.front();
  
  if(abs(groupR.left - 32) > 3){
    return;
  }
  
  for (int character_index = 0; character_index < groupR.character_rects.size(); character_index++) {
    CharacterRectListIterator rect = groupR.character_rects.begin() + character_index;
    //prepare_image_for_name_cat(card_y, as_float, rect);
    cvSetImageROI(card_y, cvRect(rect->left, rect->top, kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight));
    IplImage *temp_image = cvCreateImage (cvSize(kNameCharacterImageWidth , kNameCharacterImageHeight ), card_y->depth, card_y->nChannels);
    cvResize(card_y,temp_image);
    TensorFlowPredication pred = ApplyTensorflow::sharedInstance()->predicationFromImage(temp_image);
    cvReleaseImage(&temp_image);
    cvResetImageROI(card_y);
    newSum += pred.pValue;
    temp.push_back(pred);
    if(pred.pValue < 0.5){
      newHasUnder = true;
    }
    //dmz_debug_print("predication %c-%f.\n",pred.predication,pred.pValue);
  }
  float newStability = newSum/(float)groupR.character_rects.size();
  
  float oldStability = 0.0;
  bool oldHasUnder = false;
  if(preds.size() > 0){
    std::vector<TensorFlowPredication>::iterator it;
    float oldSum = 0.0;
    std::string fullName("");
    for(it=preds.begin() ; it < preds.end(); it++) {
      oldSum += it->pValue;
      if(it->pValue < 0.5){
        oldHasUnder = true;
      }
    }
    oldStability = oldSum / (float)preds.size();
  }
  
  dmz_debug_print("old Stability %f, new one %f.\n",oldStability, newStability);
  bool takeTheNewOne = false;
  if(newStability > oldStability){
    if(newHasUnder && !oldHasUnder && oldStability > 0.7){
      return;
    }
    if(newHasUnder && oldHasUnder){
      takeTheNewOne = true;
      preds.clear();
      preds.insert(std::end(preds), std::begin(temp), std::end(temp));
    }
    if(!newHasUnder){
      takeTheNewOne = true;
      preds.clear();
      preds.insert(std::end(preds), std::begin(temp), std::end(temp));
    }
    if(oldStability == 0.0){
      takeTheNewOne = true;
      preds.clear();
      preds.insert(std::end(preds), std::begin(temp), std::end(temp));
    }
  }else{
    if(oldHasUnder && !newHasUnder && newStability > 0.7){
      takeTheNewOne = true;
      preds.clear();
      preds.insert(std::end(preds), std::begin(temp), std::end(temp));
    }
  }
  if (takeTheNewOne) {
    
    cvSetImageROI(card_y, cvRect(groupR.left, groupR.top, groupR.width, groupR.height));
    sprintf(imageStringForName, "full-name-image.png");
    ios_save_file(imageStringForName, card_y);
    cvResetImageROI(card_y);
    
    for (int character_index = 0; character_index < groupR.character_rects.size(); character_index++) {
      CharacterRectListIterator rect = groupR.character_rects.begin() + character_index;
      cvSetImageROI(card_y, cvRect(rect->left, rect->top, kTrimmedCharacterImageWidth, kTrimmedCharacterImageHeight));
      IplImage *temp_image = cvCreateImage (cvSize(kNameCharacterImageWidth , kNameCharacterImageHeight ), card_y->depth, card_y->nChannels);
      cvResize(card_y,temp_image);
      
      sprintf(imageStringForName, "%d-name-image.png",character_index);
      ios_save_file(imageStringForName, card_y);
      
      sprintf(imageStringForName, "%d-temp-name-image.png",character_index);
      ios_save_file(imageStringForName, temp_image);
      
      cvReleaseImage(&temp_image);
      cvResetImageROI(card_y);
    }
  }
}

void scanner_add_frame_with_expiry(ScannerState *state, IplImage *y, bool scan_expiry, FrameScanResult *result) {

  bool still_need_to_collect_card_number = (state->timeOfCardNumberCompletionInMilliseconds == 0);
  bool still_need_to_scan_expiry = scan_expiry && (state->expiry_month == 0 || state->expiry_year == 0);

  // Don't bother with a bunch of assertions about y here,
  // since the frame reader will make them anyway.
  scan_card_image(y, still_need_to_collect_card_number, still_need_to_scan_expiry, result);
  if (result->upside_down) {
    return;
  }
 
  scan_analytics_record_frame(&state->session_analytics, result);

  // TODO: Scene change detection?
  
  if (!result->usable) {
    return;
  }

#if SCAN_EXPIRY
  if (still_need_to_scan_expiry) {
    state->scan_expiry = true;
    expiry_extract(y, state->expiry_groups, result->expiry_groups, &state->expiry_month, &state->expiry_year);
    //name_extract(y,state->name_groups,result->name_groups,&state->first_name,&state->second_name);
  }
#endif
  
  state->name_groups = result->name_groups;  // for now, for the debugging display
  getPredicationOfNameGroup(y,state->name_groups,state->predications);
  
  if (still_need_to_collect_card_number) {
    
    state->mostRecentUsableHSeg = result->hseg;
    state->mostRecentUsableVSeg = result->vseg;
    
    if(result->hseg.n_offsets == 15) {
      state->aggregated15 *= kDecayFactor;
      state->aggregated15 += result->scores * (1 - kDecayFactor);
      state->count15++;
    } else if(result->hseg.n_offsets == 16) {
      state->aggregated16 *= kDecayFactor;
      state->aggregated16 += result->scores * (1 - kDecayFactor);
      state->count16++;
    } else {
      assert(false);
    }
  }
}

void scanner_result(ScannerState *state, ScannerResult *result) {
  result->complete = false; // until we change our minds otherwise...avoids having to set this at all the possible early exits

#if SCAN_FOREVER
  return;
#endif

  if (state->timeOfCardNumberCompletionInMilliseconds > 0) {
    *result = state->successfulCardNumberResult;
  }
  else {
    uint16_t max_count = MAX(state->count15, state->count16);
    uint16_t min_count = MIN(state->count15, state->count16);

    // We want a three frame lead at a bare minimum.
    // Also guarantees we have at least three frames, period. :)
    if(max_count - min_count < 3) {
      return;
    }

    // Want a significant opinion about whether visa or amex
    if(min_count * 2 > max_count) {
      return;
    }
    
    result->hseg = state->mostRecentUsableHSeg;
    result->vseg = state->mostRecentUsableVSeg;

    // TODO: Sanity check the scores distributions
    // TODO: Do something else sophisticated here -- look at confidences, distributions, stability, hysteresis, etc.
    NumberScores aggregated;
    if(state->count15 > state->count16) {
      result->n_numbers = 15;
      aggregated = state->aggregated15;
    } else {
      result->n_numbers = 16;
      aggregated = state->aggregated16;
    }

    // Calculate result predictions
    // At the same time, put it in a convenient format for the basic consistency checks
    uint8_t number_as_u8s[16];

    dmz_debug_print("Stability: ");
    for(uint8_t i = 0; i < result->n_numbers; i++) {
      NumberScores::Index r, c;
      float max_score = aggregated.row(i).maxCoeff(&r, &c);
      float sum = aggregated.row(i).sum();
      result->predictions(i, 0) = c;
      number_as_u8s[i] = (uint8_t)c;
      float stability = max_score / sum;
      dmz_debug_print("%d ", (int) ceilf(stability * 100));

      // Bail early if low stability
      if (stability < kMinStability) {
        dmz_debug_print("\n");
        return;
      }
    }
    dmz_debug_print("\n");

    // Don't return a number that fails basic prefix sanity checks
    CardType card_type = dmz_card_info_for_prefix_and_length(number_as_u8s, result->n_numbers, false).card_type;
    if(card_type != CardTypeAmbiguous &&
       card_type != CardTypeUnrecognized &&
       dmz_passes_luhn_checksum(number_as_u8s, result->n_numbers)) {

      dmz_debug_print("CARD NUMBER SCANNED SUCCESSFULLY.\n");
      struct timeval time;
      gettimeofday(&time, NULL);
      state->timeOfCardNumberCompletionInMilliseconds = (long)((time.tv_sec * 1000) + (time.tv_usec / 1000));
      state->successfulCardNumberResult = *result;
    }
  }
  //HZZ MARK
  double underThreshold = 0.6;
  double statbilityThreshold = 0.9;
  
  if (counterForFailToGetName > 100){
    underThreshold = 0.0;
    statbilityThreshold = 0.5;
  }else if (counterForFailToGetName > 60){
    underThreshold = 0.3;
    statbilityThreshold = 0.5;
  }else if (counterForFailToGetName > 40){
    underThreshold = 0.4;
    statbilityThreshold = 0.65;
  }else if(counterForFailToGetName > 20){
    underThreshold = 0.4;
    statbilityThreshold = 0.7;
  }else if(counterForFailToGetName > 10){
    underThreshold = 0.5;
    statbilityThreshold = 0.8;
  }
  
  if(state->predications.size() > 0){
    std::vector<TensorFlowPredication>::iterator it;
    float sum = 0.0;
    std::string fullName("");
    bool hasUnder = false;
    for(it=state->predications.begin() ; it < state->predications.end(); it++) {
      dmz_debug_print("predication %c, %f.\n",it->predication, it->pValue);
      if(it->pValue < underThreshold){
        hasUnder = true;
      }
      sum += it->pValue;
      fullName+=it->predication;
    }
    float nStability = sum / (float)state->predications.size();
    dmz_debug_print("name predication stability %f. fail Time %d ...%f...%f\n",nStability,counterForFailToGetName,underThreshold,statbilityThreshold);
    if(nStability < statbilityThreshold || hasUnder){
      counterForFailToGetName ++;
      return;
    }else{
      counterForFailToGetName = 0.0;
      dmz_debug_print("Finsh ... name predication stability %f  result %s\n",nStability,fullName.c_str());
      state->full_name = fullName;
    }
  }else if (counterForFailToGetName > 10){
    counterForFailToGetName = 0.0;
  }else{
    return;
  }
  
  result->first_name = state->first_name;
  result->second_name = state->second_name;
  result->full_name = state->full_name;
  result->name_groups = state->name_groups;
  result->namePredications = state->predications;
  // Once the card number has been successfully scanned, then wait a bit longer for successful expiry scan (if collecting expiry)
  if (state->timeOfCardNumberCompletionInMilliseconds > 0) {
#if SCAN_EXPIRY
    if (state->scan_expiry) {
#else
    if (false) {
#endif
      struct timeval time;
      gettimeofday(&time, NULL);
      long now = (long)((time.tv_sec * 1000) + (time.tv_usec / 1000));

      if ((state->expiry_month > 0 && state->expiry_year > 0) ||
          now - state->timeOfCardNumberCompletionInMilliseconds > EXTRA_TIME_FOR_EXPIRY_IN_MICROSECONDS) {

        result->expiry_month = state->expiry_month;
        result->expiry_year = state->expiry_year;
#if DMZ_DEBUG
        result->expiry_groups = state->expiry_groups;
        //result->name_groups = state->name_groups;
#endif
        result->complete = true;

        dmz_debug_print("Extra time for expiry scan: %6.3f seconds\n", ((float)(now - state->timeOfCardNumberCompletionInMilliseconds)) / 1000.0f);
      }
    }
    else {
      result->expiry_month = 0;
      result->expiry_year = 0;
      result->complete = true;
    }
  }
}

void scanner_destroy(ScannerState *state) {
  // currently a no-op
}


#endif // COMPILE_DMZ
