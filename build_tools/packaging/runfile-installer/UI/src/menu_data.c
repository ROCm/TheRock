/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#include "menu_data.h"
#include "help_menu.h"
#include "utils.h"


int create_menu(MENU_DATA *pMenuData, WINDOW *pMenuWin, MENU_PROP *pProperties, ITEMLIST_PARAMS *pItemListParams, OFFLINE_INSTALL_CONFIG *pConfig)
{
    int subwin_numlines;

    // Set the menu window
    pMenuData->pMenuWindow = pMenuWin;
    
    pMenuData->enableMultiSelection = true;

    pMenuData->pMenuProps = pProperties;

    // Set the menu title, control message and configuration structure
    strcpy(pMenuData->menuTitle, pProperties->pMenuTitle);
    strcpy(pMenuData->menuControlMsg, pProperties->pMenuControlMsg);
    
    pMenuData->pConfig = pConfig;

    // Create items - main item list (index 0)
    add_menu_items(pMenuData, 0, pItemListParams);
    pMenuData->curItemListIndex = 0;

    // Create menu for the primary item list (0)
    pMenuData->pMenu = new_menu((ITEM**)pMenuData->itemList[0].items);

    // set the menu format
    subwin_numlines = MAX_MENU_ITEMS_DISPLAY - pProperties->starty;   // number of win lines minus control/status minus start row
    set_menu_format(pMenuData->pMenu, subwin_numlines, 1);
    pMenuData->startListIndex = 0;
    pMenuData->endListIndex = subwin_numlines;

    // set the default menu item colours (white with black backgroud)
    set_menu_fore(pMenuData->pMenu, WHITE | A_BOLD);

    // Set menu mark to the string
    set_menu_mark(pMenuData->pMenu, " > ");

    // Disable menu descriptions
    menu_opts_off(pMenuData->pMenu, O_SHOWDESC);

    // Set main window and sub window
    set_menu_win(pMenuData->pMenu, pMenuWin);
    set_menu_sub(pMenuData->pMenu, derwin(pMenuWin, subwin_numlines, pProperties->numCols,  pProperties->starty,  pProperties->startx));
    
    return 0;
}

int add_menu_items(MENU_DATA *pMenuData, int itemListIndex, ITEMLIST_PARAMS *pItemListParams)
{
    int i;
    int numItems;

    if (itemListIndex >= MAX_NUM_ITEM_LIST)
    {
        print_menu_err_msg(pMenuData, "Exceed item lists");
        return -1;
    }

    numItems = pItemListParams->numItems;

    pMenuData->itemList[itemListIndex].numItems = numItems;
    // Allocate numItems + 1 to include NULL terminator required by new_menu()
    pMenuData->itemList[itemListIndex].items = (ITEM**)calloc(numItems + 1, sizeof(ITEM*));

    int itemListTitleLen = strlen(pItemListParams->pItemListTitle);

    if (itemListTitleLen >= MAX_MENU_ITEM_COLS)
    {
        itemListTitleLen = MAX_MENU_ITEM_COLS;
    }

    strncpy(pMenuData->itemList[itemListIndex].itemTitle, pItemListParams->pItemListTitle, (size_t) itemListTitleLen);

    if (NULL != pMenuData->itemList[itemListIndex].items)
    {
        for(i = 0; i < numItems; ++i)
        {
            pMenuData->itemList[itemListIndex].items[i] = new_item(pItemListParams->pItemListChoices[i], pItemListParams->pItemListDesp[i]);
        }

        pMenuData->itemList[itemListIndex].doneItemIndex = numItems - 2;      // done item is the second-to-last item
        pMenuData->itemList[itemListIndex].helpItemIndex = numItems - 3;      // help item is the third-to-last item
    }
    else
    {
        print_menu_err_msg(pMenuData, "Failed to allocate item list");
        return -1;
    }
    
    return 0;
}

int read_file_for_items(const char *filename, char lines[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME])
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error: failed to open file %s", filename);
        return -1;
    }

    char buffer[MAX_MENU_ITEM_NAME];
    int line_count = 0;

    // Read the file line by line
    while (fgets(buffer, sizeof(buffer), file))
    {
        // Check if we've reached the maximum number of lines
        if (line_count >= MAX_MENU_ITEMS)
        {
            fprintf(stderr, "Error: Exceeded maximum number of lines (%d).\n", MAX_MENU_ITEMS);
            fclose(file);
            return -1;
        }

        // Remove the newline character, if present
        buffer[strcspn(buffer, "\n")] = '\0';

        // Copy the line into the pre-allocated array
        strncpy(lines[line_count], buffer, MAX_MENU_ITEM_NAME - 1);
        lines[line_count][MAX_MENU_ITEM_NAME - 1] = '\0'; // Ensure null termination

        line_count++;
    }

    fclose(file);
    return line_count; // Return the number of lines read
}

int read_menu_items_from_files(char *itemFile, char *itemDescFile,
                                char itemOps[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME],
                                char itemDesc[MAX_MENU_ITEMS][MAX_MENU_ITEM_NAME])
{
    int item_count, item_desc_count;

    // Read the item file
    item_count = read_file_for_items(itemFile, itemOps);
    if (item_count == -1)
    {
        exit_error("Failed to read menu items file.");
    }

    // Read the item description file
    item_desc_count = read_file_for_items(itemDescFile, itemDesc);
    if (item_desc_count == -1)
    {
        exit_error("Failed to read menu item descriptions file.");
    }

    // Validate that item count matches description count
    if (item_count != item_desc_count)
    {
        exit_error("Item count mismatch: items and descriptions files have different counts");
    }

    return item_count;
}

void destroy_menu(MENU_DATA *pMenuData)
{
    int i;
    int itemListIndex;

    // Unpost and free all the memory taken up
    unpost_menu(pMenuData->pMenu);

    //for (itemListIndex = 0; i < NUM_ITEM_LIST; ++itemListIndex)
    for (itemListIndex = 0; itemListIndex < MAX_NUM_ITEM_LIST; itemListIndex++)
    {
        if (NULL != pMenuData->itemList[itemListIndex].items)
        {
            for(i = 0; i < pMenuData->itemList[itemListIndex].numItems; ++i)
            {
                if (pMenuData->itemList[itemListIndex].items[i] != NULL)
                {
                    free_item(pMenuData->itemList[itemListIndex].items[i]);
                    pMenuData->itemList[itemListIndex].items[i] = NULL;
                }
            }
        }
    }

    free_menu(pMenuData->pMenu);

    if (NULL != pMenuData->itemList[itemListIndex].items)
    {
        free(pMenuData->itemList[itemListIndex].items);
        pMenuData->itemList[itemListIndex].items = NULL;
    }       
}

bool is_skippable_menu_item(ITEM* item)
{
    const char *name = item_name(item);

    // Original blank line check
    if (strcmp(name, SKIPPABLE_MENU_ITEM) == 0)
        return true;

    // Check for family headers (lines ending with ':' and no leading spaces)
    size_t len = strlen(name);
    if (len > 0 && name[len - 1] == ':' && name[0] != ' ')
        return true;

    return false;
}

bool skip_menu_item_down_if_skippable(MENU *pMenu)
{
    if (!is_skippable_menu_item(current_item(pMenu)))
        return false;

    ITEM *prev_item = current_item(pMenu);
    menu_driver(pMenu, REQ_DOWN_ITEM);
    ITEM *curr_item = current_item(pMenu);

    // If we didn't move (at boundary), undo by moving back up once
    if (curr_item == prev_item)
    {
        menu_driver(pMenu, REQ_UP_ITEM);
        return false;
    }

    // If we moved to another skippable item, recursively skip it
    if (is_skippable_menu_item(curr_item))
    {
        return skip_menu_item_down_if_skippable(pMenu);
    }

    return true;
}

bool skip_menu_item_up_if_skippable(MENU *pMenu)
{
    if (!is_skippable_menu_item(current_item(pMenu)))
        return false;

    ITEM *prev_item = current_item(pMenu);
    menu_driver(pMenu, REQ_UP_ITEM);
    ITEM *curr_item = current_item(pMenu);

    // If we didn't move (at boundary), undo by moving back down once
    if (curr_item == prev_item)
    {
        menu_driver(pMenu, REQ_DOWN_ITEM);
        return false;
    }

    // If we moved to another skippable item, recursively skip it
    if (is_skippable_menu_item(curr_item))
    {
        return skip_menu_item_up_if_skippable(pMenu);
    }

    return true;
}

void menu_scroll_update_selections(MENU_DATA *pMenuData, int current_index)
{
    MENU *pMenu = pMenuData->pMenu;
    ITEM **items = menu_items(pMenu);

    if (current_index >= pMenuData->endListIndex)
    {
        // delete all marked selections on the menu
        for (int i = 0; i < item_count(pMenu); i++) delete_menu_item_selection_mark(pMenuData, items[i]);

        // Always scroll by exactly 1 - the display window moves one row at a time
        // even if cursor skipped multiple items
        pMenuData->startListIndex += 1;
        pMenuData->endListIndex += 1;

        // add all marked selections for the updated locations
        for (int i = 0; i < item_count(pMenu); i++)
        {
            if (item_value(items[i])) add_menu_item_selection_mark(pMenuData, items[i]);
        }
    }
    else if (current_index < pMenuData->startListIndex)
    {
        // delete all marked selections on the menu
        for (int i = 0; i < item_count(pMenu); i++) delete_menu_item_selection_mark(pMenuData, items[i]);

        // Always scroll by exactly 1 - the display window moves one row at a time
        // even if cursor skipped multiple items
        pMenuData->startListIndex -= 1;
        pMenuData->endListIndex -= 1;

        // add all marked selections for the updated locations
        for (int i = 0; i < item_count(pMenu); i++)
        {
            if (item_value(items[i])) add_menu_item_selection_mark(pMenuData, items[i]);
        }
    }
    else if (current_index == 1 && pMenuData->startListIndex == 1 &&
             items[0] && is_skippable_menu_item(items[0]))
    {
        // Special case: Cursor at index 1, startListIndex at 1, but item 0 is skippable header
        // This means we tried to scroll up to index 0, ncurses showed it, then we skipped back
        // Display now shows index 0 at top, so adjust startListIndex and redraw marks

        // delete all marked selections
        for (int i = 0; i < item_count(pMenu); i++) delete_menu_item_selection_mark(pMenuData, items[i]);

        // Update indices to reflect that index 0 is now visible at top
        pMenuData->startListIndex = 0;
        pMenuData->endListIndex = pMenuData->startListIndex + (MAX_MENU_ITEMS_DISPLAY - pMenuData->pMenuProps->starty);

        // redraw all marked selections with new indices
        for (int i = 0; i < item_count(pMenu); i++)
        {
            if (item_value(items[i])) add_menu_item_selection_mark(pMenuData, items[i]);
        }
    }

    print_menu_scroll_info(pMenuData);
}

void print_menu_scroll_info(MENU_DATA *pMenuData)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    int totalItems = pMenuData->itemList[0].numItems - 2;  // Exclude HELP and DONE from count

    // Check if there's any content above the visible area (including skippable headers)
    bool has_content_above = (pMenuData->startListIndex > 0);

    bool has_content_below = (pMenuData->endListIndex <= totalItems);

    if (has_content_above || has_content_below)
    {
        int window_height = getmaxy(pMenuWindow);
        int window_width = getmaxx(pMenuWindow);
        int indicator_y = window_height - 5;  // Position above control message area
        int indicator_x = window_width - 8;   // Fixed position for all indicators

        wattron(pMenuWindow, BLACK_ON_WHITE | A_BOLD);

        if (has_content_above && has_content_below)
        {
            mvwprintw(pMenuWindow, indicator_y, indicator_x, " MORE ");
        }
        else if (has_content_above)
        {
            mvwprintw(pMenuWindow, indicator_y, indicator_x, "  UP  ");
        }
        else // has_content_below
        {
            mvwprintw(pMenuWindow, indicator_y, indicator_x, " DOWN ");
        }

        wattroff(pMenuWindow, BLACK_ON_WHITE | A_BOLD);
    }
    else
    {
        // Clear the indicator area if no scrolling needed
        int window_height = getmaxy(pMenuWindow);
        int window_width = getmaxx(pMenuWindow);
        int indicator_y = window_height - 5;
        int indicator_x = window_width - 8;
        mvwprintw(pMenuWindow, indicator_y, indicator_x, "      ");
    }
}

bool is_menu_item_help_item_index(MENU_DATA *pMenuData, int listIndex, ITEM *item)
{
    return pMenuData->itemList[listIndex].helpItemIndex == item_index(item);
}

bool is_menu_item_done_item_index(MENU_DATA *pMenuData, int listIndex, ITEM *item)
{
    return pMenuData->itemList[listIndex].doneItemIndex == item_index(item);
}

// Used when user selects item in menu via spacebar or enter.
void menu_item_select(MENU_DATA *pMenuData, ITEM *pCurrentItem)
{
    MENU *pMenu = pMenuData->pMenu;

    // Clears warning messages at the very bottom of the window.
    clear_menu_msg(pMenuData);

    // Deselect all other items if multiselection is disabled and the 
    // new item is not currently selected.
    if ( (!pMenuData->enableMultiSelection) && ((item_value(pCurrentItem) == FALSE)) )
    {
        ITEM **items = menu_items(pMenu);
        
        for (int i = 0; i < item_count(pMenu); i++) 
        {
            if (item_value(items[i]) == TRUE) 
            {
                set_item_value(items[i], false);
            }

            delete_menu_item_selection_mark(pMenuData, items[i]);
        }

        pMenuData->itemSelections = 0;
    }           

    // update the item selection bitfield
    TOGGLE_BIT( (pMenuData->itemSelections), (item_index(pCurrentItem)) );
    pMenuData->curItemSelection = (item_index(pCurrentItem));

    menu_driver(pMenu, REQ_TOGGLE_ITEM);

    if (item_value(pCurrentItem))
    {
        add_menu_item_selection_mark(pMenuData, pCurrentItem);
    }
    else
    {
        delete_menu_item_selection_mark(pMenuData, pCurrentItem);
    }
}

void menu_loop(MENU_DATA *pMenuData)
{
    int c;
    int done = 0;
    int listIndex = pMenuData->curItemListIndex;
    bool is_skipped = false;

    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    MENU *pMenu = pMenuData->pMenu;
    ITEM *pCurrentItem;

    void (*p)(MENU_DATA*);
    void (*drawFunc)(MENU_DATA*);
    
    // Menu loop
    while( done == 0 )
    {   
        pCurrentItem = current_item(pMenu);

        c = wgetch(pMenuWindow);
        
        switch(c)
        {	
            case KEY_RESIZE: // Terminal window resize
                if (should_window_be_resized(pMenuWindow, WIN_NUM_LINES,WIN_WIDTH_COLS))
                {
                    reset_window_before_resizing(pMenuData);
                    
                    drawFunc = pMenuData->drawMenuFunc;
                    drawFunc(pMenuData); 
                }   

                p = item_userptr(pCurrentItem);
                if (NULL != p)
                {
                    p((MENU_DATA*)pMenuData);
                }
                break;

            case KEY_DOWN:
            {
                menu_driver(pMenu, REQ_DOWN_ITEM);

                is_skipped = skip_menu_item_down_if_skippable(pMenu);

                pCurrentItem = current_item(pMenu);

                if (pMenuData->clearErrMsgAfterUpOrDownKeyPress)
                {
                    clear_menu_msg(pMenuData);
                }

                print_menu_item_selection(pMenuData, MENU_SEL_START_Y, MENU_SEL_START_X);

                // update menu scrolling position
                menu_scroll_update_selections(pMenuData, item_index(pCurrentItem));

                p = item_userptr(pCurrentItem);
                if (NULL != p)
                {
                    p((MENU_DATA*)pMenuData);
                }

                break;
            }

            case KEY_UP:
            {
                menu_driver(pMenu, REQ_UP_ITEM);

                is_skipped = skip_menu_item_up_if_skippable(pMenu);

                pCurrentItem = current_item(pMenu);

                // Special case: if we skipped at top boundary, ncurses scrolled to show index 0
                // but cursor moved back to index 1. Update startListIndex to reflect display.
                if (is_skipped && item_index(pCurrentItem) == 1 && pMenuData->startListIndex == 1)
                {
                    // Index 0 is now visible at top of display
                    pMenuData->startListIndex = 0;
                    pMenuData->endListIndex = pMenuData->startListIndex + (MAX_MENU_ITEMS_DISPLAY - pMenuData->pMenuProps->starty);

                    // Redraw all X marks with updated indices
                    ITEM **items = menu_items(pMenu);
                    for (int i = 0; i < item_count(pMenu); i++)
                    {
                        delete_menu_item_selection_mark(pMenuData, items[i]);
                    }
                    for (int i = 0; i < item_count(pMenu); i++)
                    {
                        if (item_value(items[i])) add_menu_item_selection_mark(pMenuData, items[i]);
                    }
                }

                if (pMenuData->clearErrMsgAfterUpOrDownKeyPress)
                {
                    clear_menu_msg(pMenuData);
                }

                print_menu_item_selection(pMenuData, MENU_SEL_START_Y, MENU_SEL_START_X);

                // update menu scrolling position
                menu_scroll_update_selections(pMenuData, item_index(pCurrentItem));

                p = item_userptr(pCurrentItem);
                if (NULL != p)
                {
                    p((MENU_DATA*)pMenuData);
                }

                break;
            }

            case ' ':
                // Don't do anything if item isn't selectable.
                if (item_opts(pCurrentItem) != O_SELECTABLE) continue;
                
                // Space bar selection only enabled for menus where isMenuItemSelectable is true.
                if (!pMenuData->isMenuItemsSelectable) continue;

                // Prevents users from trying to select the <DONE> and <HELP> menu items.
                if (is_menu_item_done_item_index(pMenuData, listIndex, pCurrentItem) || 
                    is_menu_item_help_item_index(pMenuData, listIndex, pCurrentItem))
                {
                    continue;   
                }

                menu_item_select(pMenuData, pCurrentItem);

                p = menu_userptr(pMenu);
                if (NULL != p)
                {
                    p((MENU_DATA*)pMenuData);
                }                
                
                break;

            case 10:    // Enter
                if ( item_index(pCurrentItem) == pMenuData->itemList[listIndex].doneItemIndex )
                {
                    done = 1;
                }
                else
                {   
                    if (is_menu_item_help_item_index(pMenuData, 0, pCurrentItem))
                    {
                        if (pMenuData->pHelpMenu)
                        {
                            // switch to the current menu's help menu
                            do_help_menu(pMenuData);

                            // switch back to current menu : redraw and post
                            menu_draw(pMenuData);
                            post_menu(pMenu);
                        }
                    }
                    else if (pMenuData->isMenuItemsSelectable && // rocm usecases or rocm versions menu
                            !is_menu_item_done_item_index(pMenuData, listIndex, pCurrentItem) &&
                            item_opts(pCurrentItem) == O_SELECTABLE ) { 

                            menu_item_select(pMenuData, pCurrentItem);
                    }

                    // call the menu data processor
                    p = menu_userptr(pMenu);
                    if (NULL != p)
                    {
                        p((MENU_DATA*)pMenuData);
                    }
                    else
                    {
                        DEBUG_UI_MSG(pMenuData, "No user ptr for form");
                    }

                    p = item_userptr(pCurrentItem);
                    if (NULL != p)
                    {
                        p((MENU_DATA*)pMenuData);
                    }
                }
                break;
        }

        wrefresh(pMenuWindow);
    }
}

void menu_draw(MENU_DATA *pMenuData)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    MENU *pMenu = pMenuData->pMenu;

    int curItemListIndex = pMenuData->curItemListIndex;

    // resizes pMenuWindow and subwindow that displays menu items to its original size in case user resized terminal window
    resize_and_reposition_window_and_subwindow(pMenuData, WIN_NUM_LINES, WIN_WIDTH_COLS);

    print_menu_title(pMenuData, MENU_TITLE_Y, MENU_TITLE_X, WIN_WIDTH_COLS, pMenuData->menuTitle, CYAN);
    print_menu_item_title(pMenuData,  ITEM_TITLE_Y, ITEM_TITLE_X, pMenuData->itemList[curItemListIndex].itemTitle, MAGENTA);
    
    print_menu_item_selection(pMenuData, MENU_SEL_START_Y, MENU_SEL_START_X);
    print_menu_control_msg(pMenuData);

    box(pMenuWindow, 0, 0);

    post_menu(pMenu);

    // redraw and currently selected items (if any)
    if (pMenuData->itemSelections)
    {
        ITEM **items = menu_items(pMenu);

        for(int i = 0; i < pMenuData->itemList[0].numItems; ++i)
        {
            if ( ((pMenuData->itemSelections) & (1 << i)) && (item_opts(items[i]) == O_SELECTABLE) )
            {
                set_item_value(items[i], true);
                add_menu_item_selection_mark(pMenuData, items[i]);
            }
        }
    }

    print_menu_scroll_info(pMenuData);

    print_version(pMenuData);
}

void menu_info_draw_bool(MENU_DATA *pMenuData, int starty, int startx, bool val)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    wmove(pMenuWindow, starty, 0);

    if (val)
    {
        wattron(pMenuWindow, GREEN);
        mvwprintw(pMenuWindow, starty, startx, "%s", "yes ");
        wattroff(pMenuWindow, GREEN);
    }
    else
    {
        wattron(pMenuWindow, RED);
        mvwprintw(pMenuWindow, starty, startx, "%s", "no ");
        wattroff(pMenuWindow, RED);
    }
}

void menu_set_item_select(MENU_DATA *pMenuData, int itemIndex, bool enable_select)
{
    if (enable_select)
    {
        item_opts_on(pMenuData->itemList[0].items[itemIndex], O_SELECTABLE);
    }
    else
    {
        item_opts_off(pMenuData->itemList[0].items[itemIndex], O_SELECTABLE);
    }
}

void print_menu_title(MENU_DATA *pMenuData, int starty, int startx, int width, char *string, chtype color)
{
    int length, x, y;
    float temp;

    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    getyx(pMenuWindow, y, x);
    if(startx != 0)
    {
        x = startx;
    }

    if(starty != 0)
    {
        y = starty;
    }

    if(width == 0)
    {
        width = WIN_WIDTH_COLS;
    }

    length = strlen(string);
    temp = (width - length)/ 2;

    x = startx + (int)temp;

    wmove(pMenuWindow, y, 0);
    wclrtoeol(pMenuWindow);

    mvwhline(pMenuWindow, 2, 2, ACS_HLINE, WIN_WIDTH_COLS - 4);

    wattron(pMenuWindow, color | A_BOLD);
    mvwprintw(pMenuWindow, y, x, "%s", string);
    wattroff(pMenuWindow, color | A_BOLD);
}

void print_menu_item_title(MENU_DATA *pMenuData, int starty, int startx, char *string, chtype color)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    wmove(pMenuWindow, starty, 0);
    wclrtoeol(pMenuWindow);

    wattron(pMenuWindow, color);
    mvwprintw(pMenuWindow, starty, startx, "%s", string);
    wattroff(pMenuWindow, color);
}

void print_border_around_item_description(WINDOW *pMenuWindow, int starty)
{   
    mvwaddch(pMenuWindow, starty, WIN_WIDTH_COLS-1, ACS_VLINE);
    mvwaddch(pMenuWindow, starty+1, 0, ACS_VLINE);
    mvwaddch(pMenuWindow, starty+1, WIN_WIDTH_COLS-1, ACS_VLINE);
}

void remove_menu_item_selection_description(MENU_DATA *pMenuData, int starty, int startx)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    wmove(pMenuWindow, starty, startx);
    wclrtoeol(pMenuWindow);

    wmove(pMenuWindow, starty+1, 0);
    wclrtoeol(pMenuWindow);

    print_border_around_item_description(pMenuWindow, starty);
}

void print_menu_item_selection_opt(MENU_DATA *pMenuData, int starty, int startx, const char *description)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    
    wmove(pMenuWindow, starty, startx);
    wclrtoeol(pMenuWindow);

    wmove(pMenuWindow, starty+1, 0);
    wclrtoeol(pMenuWindow);

#if ENABLE_MENU_DEBUG
    MENU *pMenu = pMenuData->pMenu;
    ITEM *pItem = current_item(pMenu);
    mvwprintw(pMenuWindow, starty, startx, "%d: name %s: %s", 
        item_index(pItem) + 1,  
        item_name(pItem), 
        description);
#else
    mvwprintw(pMenuWindow, starty, startx, "* %s", description);
#endif // ENABLE_MENU_DEBUG

    print_border_around_item_description(pMenuWindow, starty);
}

void print_menu_item_selection(MENU_DATA *pMenuData, int starty, int startx)
{
    ITEM *pItem = current_item(pMenuData->pMenu);
    
    // draw the item description if present
    if (item_description(pItem))
    {
        print_menu_item_selection_opt(pMenuData, starty, startx, item_description(pItem));
    }
}

void print_version(MENU_DATA *pMenuData)
{
    // Use runtime-loaded version instead of compile-time macros
    OFFLINE_INSTALL_CONFIG *pConfig = pMenuData->pConfig;
    if (pConfig != NULL)
    {
        mvwprintw(pMenuData->pMenuWindow, 28, 70, "v%s-%s", pConfig->installerVersion, pConfig->rocmVersion);
    }
}

void print_menu_msg(MENU_DATA *pMenuData, chtype color, const char *fmt, ...)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    int x = DEBUG_ERR_START_X;
    int y = DEBUG_ERR_START_Y;

    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(NULL, 0, fmt, args);
    va_end(args);
    if(len < 0) return;

    // format the string
    char string[len + 1];

    va_start(args, fmt);
    vsnprintf(string, len + 1, fmt, args);
    va_end(args);

    wmove(pMenuWindow, y, x);
    wclrtoeol(pMenuWindow);
    
    wattron(pMenuWindow, color | A_BOLD);
    mvwprintw(pMenuWindow, y, x, "%s", string);
    wattroff(pMenuWindow, color | A_BOLD);

    print_border_around_item_description(pMenuWindow, y-1);
    print_version(pMenuData);
}

void print_menu_warning_msg(MENU_DATA *pMenuData, const char *fmt, ...)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    int x = DEBUG_ERR_START_X;
    int y = DEBUG_ERR_START_Y;

    va_list args;
    va_start(args, fmt);
    
    int len = vsnprintf(NULL, 0, fmt, args);
    va_end(args);

    if(len < 0) return;

    // format the string
    char string[len + 1];

    va_start(args, fmt);
    vsnprintf(string, len + 1, fmt, args);
    va_end(args);

    wmove(pMenuWindow, y, x);
    wclrtoeol(pMenuWindow);
    wattron(pMenuWindow, YELLOW | A_BOLD);
    mvwprintw(pMenuWindow, y, x, "WARNING: %s", string);
    wattroff(pMenuWindow, YELLOW | A_BOLD);

    print_border_around_item_description(pMenuWindow, y-1);
    print_version(pMenuData);
}

void print_menu_err_msg(MENU_DATA *pMenuData, const char *fmt, ...)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    int x = DEBUG_ERR_START_X;
    int y = DEBUG_ERR_START_Y;

    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(NULL, 0, fmt, args);
    va_end(args);
    if(len < 0) return;

    // format the string
    char string[len + 1];

    va_start(args, fmt);
    vsnprintf(string, len + 1, fmt, args);
    va_end(args);

    wmove(pMenuWindow, y, x);
    wclrtoeol(pMenuWindow);
    
    wattron(pMenuWindow, RED | A_BOLD);
    mvwprintw(pMenuWindow, y, x, "ERROR: %s", string);
    wattroff(pMenuWindow, RED | A_BOLD);

    print_border_around_item_description(pMenuWindow, y-1);
    print_version(pMenuData);
}

void clear_menu_msg(MENU_DATA *pMenuData)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    
    wmove(pMenuWindow, DEBUG_ERR_START_Y, DEBUG_ERR_START_X);
    wclrtoeol(pMenuWindow);
    print_border_around_item_description(pMenuWindow, DEBUG_ERR_START_Y-1);
    print_version(pMenuData);
}

void print_menu_dbg_msg(MENU_DATA *pMenuData, const char *fmt, ...)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;

    int x = DEBUG_ERR_START_X;
    int y = DEBUG_ERR_START_Y;

    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(NULL, 0, fmt, args);
    va_end(args);
    if(len < 0) return;

    // format the string
    char string[len + 1];

    va_start(args, fmt);
    vsnprintf(string, len + 1, fmt, args);
    va_end(args);

    wmove(pMenuWindow, y, x);
    wclrtoeol(pMenuWindow);

    mvwprintw(pMenuWindow, y, x, "DEBUG: %s", string);
}

void print_menu_control_msg(MENU_DATA *pMenuData)
{
    move(WIN_NUM_LINES + 5, 0);
    clrtoeol();

    mvprintw(WIN_NUM_LINES + 5, 0, "%s", pMenuData->menuControlMsg);
    refresh();
}

int create_form(MENU_DATA *pMenuData, WINDOW *pMenuWin, int numFields, int width, int height, int starty, int startx)
{ 
    int i;
    int rows, cols;

    if (numFields >= MAX_NUM_FORM_FIELDS)
    {
        return -1;
    }

    for (i = 0; i < numFields; i++)
    {
         //                                                        row,     col
        pMenuData->pFormList.field[i] = new_field(height, width, starty+i,  startx,  0, 0);     // initial line-by-line (can be moved)

        field_opts_off(pMenuData->pFormList.field[i], O_AUTOSKIP);  // don't skip to the next field with filled
        field_opts_off(pMenuData->pFormList.field[i], O_BLANK);     // don't delete entire field if a character is added in the first position

        field_opts_off(pMenuData->pFormList.field[i], O_STATIC);             // enable dynamic fields
        set_max_field(pMenuData->pFormList.field[i], MAX_FORM_FIELD_WIDTH);  // set the max field width for the dynamic field
    }

    pMenuData->pFormList.pForm = new_form(pMenuData->pFormList.field);
    pMenuData->pFormList.numFields = numFields;

    scale_form(pMenuData->pFormList.pForm, &rows, &cols);

    set_form_win(pMenuData->pFormList.pForm, pMenuWin);
    set_form_sub(pMenuData->pFormList.pForm, derwin(pMenuWin,  rows, cols, 0, 0));

    return 0;
}

void destroy_form(MENU_DATA *pMenuData)
{
    int i;

    unpost_form(pMenuData->pFormList.pForm);

    for (i = 0; i < pMenuData->pFormList.numFields; i++)
    {
        free_field(pMenuData->pFormList.field[i]);
    }

    free_form(pMenuData->pFormList.pForm);
}

void form_loop(MENU_DATA *pMenuData, bool enableNextField)
{
    int c, x;
    int done = 0;

    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    FORM *pForm = pMenuData->pFormList.pForm;

    curs_set(1);

    // move to the end of the field (default values if already set)
    form_driver(pMenuData->pFormList.pForm, REQ_END_LINE);

    wrefresh(pMenuWindow);

    // Form loop
    while( done == 0 )
    {
        c = wgetch(pMenuWindow);

        DEBUG_UI_MSG(pMenuData, "Hit : %c or %d : currentFieldIndex %d", c, c, field_index(current_field(pForm)));
        switch(c)
	    {
            case KEY_DOWN:
                if (enableNextField)
                {
                    // Go to next field
                    form_driver(pForm, REQ_NEXT_FIELD);
                    form_driver(pForm, REQ_END_LINE);
                }
                break;

	        case KEY_UP:
                if (enableNextField)
                {
                    // Go to previous field
                    form_driver(pForm, REQ_PREV_FIELD);
                    form_driver(pForm, REQ_END_LINE);
                }
                break;

            case KEY_BACKSPACE:
            case 8:
            case 127:
                x = getcurx(pMenuWindow);
                
                FIELD *pCurrentField = current_field(pForm);
                // don't run REQ_DEL_PREV if cursor is at the beginning of the field
                // due to ncurses behaviour that changes cmd REQ_DEL_PREV into
                // REQ_PREV_FIELD when on the first char.
                // workaround of setting option O_BS_OVERLOAD off doesn't work.
                if (x > pCurrentField->fcol ) 
                {
                    form_driver(pForm, REQ_DEL_PREV);
                }

                break;

            case CTRL('D'):
                DEBUG_UI_MSG(pMenuData, "Ctrl-D");
                form_driver(pForm, REQ_CLR_FIELD);
                break;

            case 10: //Enter
                form_driver(pForm, REQ_NEXT_FIELD);
                form_driver(pForm, REQ_PREV_FIELD);

                done = 1;

                break;
                
            case KEY_DC:
                form_driver(pForm, REQ_DEL_CHAR);
                break;
                
            case KEY_LEFT:
                form_driver(pForm, REQ_LEFT_CHAR);
                break;
                
            case KEY_RIGHT:
                form_driver(pForm, REQ_RIGHT_CHAR);
                break;
                
            default:	
                form_driver(pForm, c);
                break;
        }
    }

    curs_set(0);
}

void print_form_control_msg(MENU_DATA *pMenuData)
{
    move(WIN_NUM_LINES + 5, 0);
    clrtoeol();

    mvprintw(WIN_NUM_LINES + 5, 0, "%s", pMenuData->pFormList.formControlMsg);
    refresh();
}

bool should_window_be_resized(WINDOW *pMenuWindow, int y, int x)
{
    int menuWindowX, menuWindowY;
    getmaxyx(pMenuWindow, menuWindowY, menuWindowX);

    if (menuWindowX < x || menuWindowY < y)
    {
        return false;
    }

    return true;
}

void resize_and_reposition_window_and_subwindow(MENU_DATA *pMenuData, int y, int x)
{
    WINDOW *pMenuWindow = pMenuData->pMenuWindow;
    MENU_PROP *pProperties = pMenuData->pMenuProps;
    WINDOW *pSubMenuWindow = menu_sub(pMenuData->pMenu); // displays menu items

    wresize(pMenuWindow, y, x);
    
    // resize and resposition pSubMenuWindow relative to parent window
    wresize(pSubMenuWindow, pProperties->numLines, pProperties->numCols);
    mvderwin(pSubMenuWindow, pProperties->starty, pProperties->startx);
}

void reset_window_before_resizing(MENU_DATA *pMenuData)
{
    unpost_menu(pMenuData->pMenu);
    clear();
    endwin();
}

void add_menu_item_selection_mark(MENU_DATA *pMenuData, ITEM *pCurrentItem)
{
    WINDOW *pSubMenuWindow = menu_sub(pMenuData->pMenu);

    int itemIndex = item_index(pCurrentItem);
    int actual_top_row = top_row(pMenuData->pMenu);
    int index = itemIndex - actual_top_row;
    int y = getpary(pSubMenuWindow);

    // Only draw if item is in the visible area
    if (index >= 0 && index < (pMenuData->endListIndex - pMenuData->startListIndex))
    {
        mvwprintw(pMenuData->pMenuWindow, y + index, 2, "X");
    }
}

void delete_menu_item_selection_mark(MENU_DATA *pMenuData, ITEM *pCurrentItem)
{
    WINDOW *pSubMenuWindow = menu_sub(pMenuData->pMenu);

    int itemIndex = item_index(pCurrentItem);

    // Use ncurses' own top_row to get actual displayed top row
    int actual_top_row = top_row(pMenuData->pMenu);
    int index = itemIndex - actual_top_row;
    int y = getpary(pSubMenuWindow);

    mvwprintw(pMenuData->pMenuWindow, y + index, 2, " ");
}

/*
 * Display help text from a file in a scrollable window.
 *
 * Supports heading tags for formatted text display:
 *
 * Color Scheme Table:
 * +---------+-----------+------------+---------------------------+
 * | Tag     | Color     | Attributes | Use Case                  |
 * +---------+-----------+------------+---------------------------+
 * | [H1]    | Cyan      | Bold       | Main page title           |
 * | [H2]    | Magenta   | Bold       | Major section headers     |
 * | [H3]    | Yellow    | Bold       | Subsection headers, items |
 * +---------+-----------+------------+---------------------------+
 *
 * Usage in help files:
 *   [H1]Page Title
 *   Introduction text...
 *
 *   [H2]Section Header
 *   Section content...
 *
 *   [H3]Subsection Header
 *     Detailed information...
 *
 *   Regular text without formatting.
 *
 * Tags are automatically stripped from display output.
 */
int display_help_scroll_window(MENU_DATA *pMenuData, char *filename)
{
    WINDOW *win;

    int done = 0;

    char **lines = NULL;
    char *line = NULL;
    size_t line_length = 0;
    ssize_t read;

    int start_y = ITEM_TITLE_Y;
    int start_line = 0;
    int num_lines = start_y;
    int c;

    int max_win_lines = WIN_NUM_LINES - 4;

    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        print_menu_err_msg(pMenuData, "Failed to open file %s. Press any key to exit this menu.", filename);
        return -1;
    }

    win = pMenuData->pMenuWindow;

    scrollok(win, TRUE);
    keypad(win, TRUE);
    curs_set(0);

    // Read in all the lines making up the file
    while ((read = getline(&line, &line_length, file)) != -1) 
    {
        char **temp = realloc(lines, (num_lines + 1) * sizeof(char*));
        if (temp == NULL)
        {
            if (line != NULL) free(line);

            if (lines != NULL)
            {
                for (int i = 0; i < num_lines; i++)
                {
                    if (lines[i] != NULL) free(lines[i]);
                }
                free(lines);
            }
            
            fclose(file);
            return 1;
        }

        lines = temp;
        lines[num_lines] = line;

        line = NULL;
        line_length = 0;
        
        num_lines++;
    }

    free(line);
    fclose(file);

    // adjust for single page files
    if (num_lines < max_win_lines)
    {
        max_win_lines = num_lines + 1;
    }

    // Draw the contents of the file line-by-line
    while( done == 0 )
    {
        // Display all the lines of text up until the size of the window
        for (int i = start_y; i < max_win_lines - 1; i++)
        {
            if (lines != NULL)
            {
                if (lines[start_line + i] != NULL)
                {
                    char *line_text = lines[start_line + i];
                    char *display_text = line_text;
                    int heading_type = 0;

                    // Check for heading tags [H1], [H2], [H3]
                    if (strncmp(line_text, "[H1]", 4) == 0)
                    {
                        heading_type = 1;
                        display_text = line_text + 4;  // Skip the tag
                    }
                    else if (strncmp(line_text, "[H2]", 4) == 0)
                    {
                        heading_type = 2;
                        display_text = line_text + 4;  // Skip the tag
                    }
                    else if (strncmp(line_text, "[H3]", 4) == 0)
                    {
                        heading_type = 3;
                        display_text = line_text + 4;  // Skip the tag
                    }

                    // Apply formatting based on heading type
                    if (heading_type == 1)
                    {
                        wattron(win, CYAN | A_BOLD);  // Heading 1: Cyan + Bold
                        mvwprintw(win, i+1, 1, display_text, "%s");
                        wattroff(win, CYAN | A_BOLD);
                    }
                    else if (heading_type == 2)
                    {
                        wattron(win, MAGENTA | A_BOLD);  // Heading 2: Magenta + Bold
                        mvwprintw(win, i+1, 1, display_text, "%s");
                        wattroff(win, MAGENTA | A_BOLD);
                    }
                    else if (heading_type == 3)
                    {
                        wattron(win, YELLOW | A_BOLD);  // Heading 3: Yellow + Bold
                        mvwprintw(win, i+1, 1, display_text, "%s");
                        wattroff(win, YELLOW | A_BOLD);
                    }
                    else
                    {
                        mvwprintw(win, i+1, 1, line_text, "%s");
                    }
                }
            }
        }

        box(win, 0, 0);

        // Draw scroll indicators at bottom right corner (inside the box)
        int window_height = getmaxy(win);
        int window_width = getmaxx(win);

        // Check if there's more content above or below
        bool has_content_above = (start_line > 0);
        bool has_content_below = (start_line < num_lines - (max_win_lines - 1));

        if (has_content_above || has_content_below)
        {
            int indicator_y = window_height - 2;  // 1 line above bottom border
            int indicator_x = window_width - 8;   // Fixed position for all indicators

            wattron(win, BLACK_ON_WHITE | A_BOLD);

            if (has_content_above && has_content_below)
            {
                mvwprintw(win, indicator_y, indicator_x, " MORE ");
            }
            else if (has_content_above)
            {
                mvwprintw(win, indicator_y, indicator_x, "  UP  ");
            }
            else // has_content_below
            {
                mvwprintw(win, indicator_y, indicator_x, " DOWN ");
            }

            wattroff(win, BLACK_ON_WHITE | A_BOLD);
        }

        wrefresh(win);

        c = wgetch(win);
        switch (c) 
        {
            case KEY_RESIZE:
                if (should_window_be_resized(win, WIN_NUM_LINES,WIN_WIDTH_COLS))
                {
                    wclear(win);
                    menu_draw(pMenuData);
                }
                break;
            case KEY_DOWN:
                if (start_line < num_lines - (max_win_lines-1))
                {
                    start_line++;
                }
                break;

            case KEY_UP:
                if (start_line > 0) 
                {
                    start_line--;
                }
                break;

            default:
                done = 1;
                break;
        }
    };

    if (lines != NULL)
    {
        for (int i = start_y; i < num_lines; i++)
        {
            if (lines[i] != NULL) free(lines[i]);
        }
        free(lines);
    }

    return 0;
}

int display_scroll_window(char *windowTitle, char *listTitle, char *filename, int *pNumLines)
{
    WINDOW *win;

    int done = 0;

    char **lines = NULL;
    char *line = NULL;
    size_t line_length = 0;
    ssize_t read;

    int start_line = 0;
    int num_lines = 0;
    int c;
    float temp;

    int list_start_row = 5;   // starting line for item list

    int max_win_lines, currentLine;

    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        return -1;
    }

    win = newwin(WIN_NUM_LINES, WIN_WIDTH_COLS, WIN_START_Y, WIN_START_X);

    scrollok(win, TRUE);
    keypad(win, TRUE);

    max_win_lines = WIN_NUM_LINES - list_start_row;
    currentLine = max_win_lines;

    // Draw the window title
    temp = (WIN_WIDTH_COLS - strlen(windowTitle))/ 2;
    
    wattron(win, CYAN | A_BOLD);
    mvwprintw(win, 1, (int)temp, "%s", windowTitle);
    wattroff(win, CYAN | A_BOLD);

    mvwhline(win, 2, 2, ACS_HLINE, WIN_WIDTH_COLS - 4);

    // Draw the secondary title for the list
    if (listTitle)
    {
        wattron(win, MAGENTA | A_BOLD);
        mvwprintw(win, 3, 10, "%s", listTitle);
        wattroff(win, MAGENTA | A_BOLD);
    }

    // Draw the control msg
    move(WIN_NUM_LINES + 5, 0);
    clrtoeol();
    mvprintw(WIN_NUM_LINES + 5, 0, "%s", "<UP/DOWN> To scroll | any other key to exit");
    refresh();

    // Read in all the lines making up the file
    while ((read = getline(&line, &line_length, file)) != -1) 
    {
        char **temp = realloc(lines, (num_lines + 1) * sizeof(char*));
        if (temp == NULL)
        {
            if (line != NULL) free(line);

            if (lines != NULL)
            {
                for (int i = 0; i < num_lines; i++)
                {
                    if (lines[i] != NULL) free(lines[i]);
                }
                free(lines);
            }

            fclose(file);
            return 1;
        }

        lines = temp;
        lines[num_lines] = line;

        line = NULL;
        line_length = 0;
        
        num_lines++;
    }

    free(line);
    fclose(file);

    // adjust for single page files
    if (num_lines < max_win_lines)
    {
        max_win_lines = num_lines + 1;
    }

    // Draw the total number of lines for the scroll window
    wattron(win, BLACK_ON_MAGENTA | A_BOLD);
    mvwprintw(win, 3, 1, "Total %d", num_lines);
    wattroff(win, BLACK_ON_MAGENTA | A_BOLD);
    
    // Draw the contents of the file line-by-line
    while( done == 0 )
    {
        // Display all the lines of text up until the size of the window
        for (int i = 0; i < max_win_lines - 1; i++) 
        {
            if (lines != NULL)
            {
                if (lines[start_line + i] != NULL)
                {
                    mvwprintw(win, i+list_start_row, 1, lines[start_line + i], "%s");
                }
            }       
        }

        // draw the line count message
        currentLine = max_win_lines + start_line - 1;
        if (num_lines > max_win_lines)
        {
            move(WIN_NUM_LINES + 4, 0);
            clrtoeol();
            attron(BLACK_ON_WHITE | A_BOLD);
            mvprintw(WIN_NUM_LINES + 4, 4, "<< %d, %d >>", currentLine, num_lines);
            attroff(BLACK_ON_WHITE | A_BOLD);
            refresh();
        }

        box(win, 0, 0);

        wrefresh(win);

        c = wgetch(win);
        switch (c) 
        {
            case KEY_DOWN:
                if (start_line < num_lines - (max_win_lines-1))
                {
                    start_line++;
                }
                break;

            case KEY_UP:
                if (start_line > 0) 
                {
                    start_line--;
                }
                break;

            default:
                done = 1;
                break;
        }
    };

    delwin(win);

    // return the number of lines for the scroll window
    if (pNumLines)
    {
        *pNumLines = num_lines;
    }

    // clean up
    if (lines != NULL)
    {
        for (int i = 0; i < num_lines; i++) 
        {
            if (lines[i] != NULL) free(lines[i]);
        }
        free(lines);
    }

    // clear the control area
    move(WIN_NUM_LINES + 4, 0);
    clrtoeol();

    return 0;
}

void draw_progress_bar(WINDOW *win, int percentage, int show) 
{
    int bar_width = (percentage * PROGRESS_BAR_WIDTH) / 100;
    mvwprintw(win, 2, 1, "[");

    for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) 
    {
        if (i < bar_width) 
        {
            waddch(win, '=');
        } 
        else 
        {
            waddch(win, ' ');
        }
    }

    wprintw(win, "]"); 

    if (show)
    {
        mvwprintw(win, 2, 1, "%3d%%", percentage);
    }
   
    wrefresh(win);
}

int wait_with_progress_bar(pid_t pid, int time, int show)
{
    int height = 3; 
    int width = PROGRESS_BAR_WIDTH + 5;
    int start_y = WIN_NUM_LINES;
    int start_x = WIN_START_X + 1;
    
    WINDOW *progress_win = newwin(height, width, start_y, start_x);
    wrefresh(progress_win);

    int status;
    int progress = 0;
    while (waitpid(pid, &status, WNOHANG) == 0) 
    {
        draw_progress_bar(progress_win, progress, show);
        progress = (progress + 1) % 101; // Loop progress from 0 to 100

        usleep(time);
    }

    return (WEXITSTATUS(status));
}

