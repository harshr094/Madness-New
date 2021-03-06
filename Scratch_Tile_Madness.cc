#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath> 
#include <cstdio>
#include "legion.h"
#include <vector>
#include <queue>
#include <utility>

using namespace Legion;
using namespace std;

enum TASK_IDs
{
    TOP_LEVEL_TASK_ID,
    REFINE_INTER_TASK_ID,
    REFINE_INTRA_TASK_ID,
    PRINT_TASK_ID,
    COMPRESS_INTER_TASK_ID,
    COMPRESS_INTRA_TASK_ID,
    COMPRESS_UPDATE_TASK_ID,
    RECONSTRUCT_INTER_TASK_ID,
    RECONSTRUCT_INTRA_TASK_ID,
    NORM_INTER_TASK_ID,
    NORM_INTRA_TASK_ID,
    INNER_PRODUCT_INTER_TASK_ID,
    INNER_PRODUCT_INTRA_TASK_ID,
    GAXPY_INTER_TASK_ID,
    GAXPY_INTRA_TASK_ID
};

enum FieldId{
    FID_X,
};

struct Arguments {
    int n;
    int l;
    int actual_l;
    int max_depth;
    coord_t idx;
    coord_t end_idx;
    long int gen;
    Color partition_color;
    int actual_max_depth;
    int tile_height;
    int root_location;
    int carry;
    Arguments(int _n, int _l, int _actual_l , int _max_depth, coord_t _idx, coord_t _end_idx, Color _partition_color, int _actual_max_depth=0, int _tile_height=1, int _root_location=1, int _carry =0 )
        : n(_n), l(_l), actual_l(_actual_l), max_depth(_max_depth), idx(_idx), end_idx(_end_idx), partition_color(_partition_color), actual_max_depth(_actual_max_depth), tile_height(_tile_height), root_location(_root_location),carry(_carry)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct InnerProductArgs{
    int n;
    int l;
    int max_depth;
    coord_t idx;
    coord_t end_idx;
    long int gen;
    Color partition_color1, partition_color2;
    int actual_max_depth;
    int tile_height;
    InnerProductArgs(int _n, int _l, int _max_depth, coord_t _idx, coord_t _end_idx, Color _partition_color1, Color _partition_color2, int _actual_max_depth=0, int _tile_height=1 )
        : n(_n), l(_l), max_depth(_max_depth), idx(_idx), end_idx(_end_idx) ,partition_color1(_partition_color1), partition_color2(_partition_color2), actual_max_depth(_actual_max_depth), tile_height(_tile_height)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct GaxpyArgs{
    int n;
    int l;
    int actual_l;
    int max_depth;
    coord_t idx;
    coord_t end_idx;
    long int gen;
    Color partition_color1, partition_color2, partition_color3;
    int pass;
    int actual_max_depth;
    int tile_height;
    bool left_null, right_null;
    GaxpyArgs(int _n, int _l, int _actual_l, int _max_depth, coord_t _idx, coord_t _end_idx, Color _partition_color1, Color _partition_color2, Color _partition_color3, int _pass, bool _left_null, bool _right_null, int _actual_max_depth=0, int _tile_height=1 )
        : n(_n), l(_l), actual_l(_actual_l) , max_depth(_max_depth), idx(_idx), end_idx(_end_idx),partition_color1(_partition_color1), partition_color2(_partition_color2), partition_color3(_partition_color3) ,pass(_pass), left_null(_left_null), right_null(_right_null), actual_max_depth(_actual_max_depth), tile_height(_tile_height)
    {
        if (_actual_max_depth == 0) {
            actual_max_depth = _max_depth;
        }
    }
};

struct TreeArgs{
    int value;
    int lval;
    bool is_leaf;
    TreeArgs( int _value, int _lval , bool _is_leaf=false ) : value(_value), lval(_lval), is_leaf(_is_leaf) {}
};

struct GaxpyHelper{
    int n,l;
    coord_t idx;
    int pass;
    bool left_null, right_null;
    bool launch;
    int actual_l;
    GaxpyHelper( int _n, int _l, coord_t _idx, int _pass, bool _left_null, bool _right_null , bool _launch, int _actual_l ) : n(_n), l(_l), idx(_idx), pass(_pass), left_null(_left_null), right_null(_right_null), launch(_launch), actual_l(_actual_l)    {}
};

struct HelperArgs{
    int level;
    int actual_l;
    coord_t idx;
    bool launch;
    int n;
    bool is_valid_entry;
    int carry;
    HelperArgs( int _level, int _actual_l ,coord_t _idx, bool _launch, int _n , bool _is_valid_entry=false, int _carry = 0 ) : level(_level), actual_l(_actual_l) ,idx(_idx), launch(_launch), n(_n), is_valid_entry( _is_valid_entry ), carry(_carry) {}
};

struct RootPosArgs{
    int value;
    RootPosArgs( int _value =1 ): value(_value) {}
 };



void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime) {

    int overall_max_depth = 7;
    int actual_left_depth = 0;
    int tile_height = 3;

    long int seed = 12345;
    {
        const InputArgs &command_args = HighLevelRuntime::get_input_args();
        for (int idx = 1; idx < command_args.argc; ++idx)
        {
            if (strcmp(command_args.argv[idx], "-max_depth") == 0)
                overall_max_depth = atoi(command_args.argv[++idx]);
            else if (strcmp(command_args.argv[idx], "-seed") == 0)
                seed = atol(command_args.argv[++idx]);
            else if(strcmp(command_args.argv[idx],"--tile") == 0)
                tile_height = atoi( command_args.argv[++idx]);
        }
    }
    srand(time(NULL));
    Rect<1> tree_rect(0LL, static_cast<coord_t>(pow(2, overall_max_depth)));
    IndexSpace is = runtime->create_index_space(ctx, tree_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(TreeArgs), FID_X);
    }
    LogicalRegion lr1 = runtime->create_logical_region(ctx, is, fs);
    Color partition_color1 = 10;
    coord_t end_idx = (1<<overall_max_depth)-1;
    Arguments args1(0, 0, 0, overall_max_depth, 0, end_idx, partition_color1, actual_left_depth, tile_height);
    args1.gen = rand();
    cout<<"Launching Refine Task"<<endl;
    TaskLauncher refine_launcher(REFINE_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    refine_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    refine_launcher.add_field(0, FID_X);
    runtime->execute_task(ctx, refine_launcher);

    cout<<"Launching Print Task After Refine"<<endl;
    TaskLauncher print_launcher(PRINT_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    RegionRequirement req3( lr1 , READ_ONLY, EXCLUSIVE, lr1 );
    req3.add_field(FID_X);
    print_launcher.add_region_requirement( req3 );
    runtime->execute_task(ctx, print_launcher);

    // cout<<"Launching Compress Task"<<endl;
    // Rect<1> root_location(0, 1);
    // is = runtime->create_index_space(ctx, root_location);
    // fs = runtime->create_field_space(ctx);
    // {
    //     FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    //     allocator.allocate_field(sizeof(RootPosArgs), FID_X);
    // }
    // LogicalRegion root_locate_region = runtime->create_logical_region(ctx, is, fs);
    // args1.root_location=-1;
    // TaskLauncher compress_launcher(COMPRESS_INTER_TASK_ID, TaskArgument(&args1, sizeof(Arguments)));
    // compress_launcher.add_region_requirement(RegionRequirement(lr1, WRITE_DISCARD, EXCLUSIVE, lr1));
    // compress_launcher.add_region_requirement(RegionRequirement(root_locate_region,WRITE_DISCARD,EXCLUSIVE,root_locate_region));
    // compress_launcher.add_field(0,FID_X);
    // compress_launcher.add_field(1,FID_X);
    // runtime->execute_task(ctx, compress_launcher);

    // cout<<"Launching Print After Compress"<<endl;
    // runtime->execute_task(ctx,print_launcher);
    // cout<<"Launching Reconstruct Task"<<endl;
    // args1.carry=0;
    // TaskLauncher reconstruct_launcher(RECONSTRUCT_INTER_TASK_ID, TaskArgument(&args1,sizeof(Arguments)));
    // reconstruct_launcher.add_region_requirement(RegionRequirement(lr1,WRITE_DISCARD,EXCLUSIVE,lr1));
    // reconstruct_launcher.add_field(0,FID_X);
    // runtime->execute_task(ctx, reconstruct_launcher);

    // cout<<"Launching Print After Reconstruct"<<endl;
    // runtime->execute_task(ctx,print_launcher);

    // cout<<"Launching Norm Task"<<endl;
    // TaskLauncher norm_launcher(NORM_INTER_TASK_ID, TaskArgument(&args1,sizeof(Arguments)));
    // norm_launcher.add_region_requirement(RegionRequirement(lr1,WRITE_DISCARD,EXCLUSIVE,lr1));
    // norm_launcher.add_field(0,FID_X);
    // Future f = runtime->execute_task(ctx,norm_launcher);
    // cout<<sqrt(f.get_result<int>())<<endl;

    cout<<"Creating 2nd Logical Region "<<overall_max_depth<<endl;
    Rect<1> tree_second(0LL, static_cast<coord_t>(pow(2, overall_max_depth)));
    IndexSpace is2 = runtime->create_index_space(ctx, tree_second);
    FieldSpace fs2 = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs2);
        allocator.allocate_field(sizeof(TreeArgs), FID_X);
    }
    LogicalRegion lr2 = runtime->create_logical_region(ctx, is2, fs2);
    Color partition_color2 = 20;
    Arguments args2(0, 0, 0,overall_max_depth, 0, end_idx, partition_color2, actual_left_depth, tile_height);
    args2.gen=rand();
    //cout<<"Launching Refine Task For 2nd  Tree"<<endl;
    TaskLauncher refine_launcher2(REFINE_INTER_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    refine_launcher2.add_region_requirement(RegionRequirement(lr2, WRITE_DISCARD, EXCLUSIVE, lr2));
    refine_launcher2.add_field(0, FID_X);
    runtime->execute_task(ctx, refine_launcher2);

    //cout<<"Print Task for 2nd Tree"<<endl;
    TaskLauncher print_launcher2(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    RegionRequirement req4( lr2 , READ_ONLY, EXCLUSIVE, lr2 );
    req4.add_field(FID_X);
    print_launcher2.add_region_requirement( req4 );
    runtime->execute_task(ctx, print_launcher2);
    
    // cout<<"Launching Compress Task for 2nd Tree"<<endl;
    // TaskLauncher compress_launcher2(COMPRESS_INTER_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // compress_launcher2.add_region_requirement(RegionRequirement(lr2, WRITE_DISCARD, EXCLUSIVE, lr2));
    // compress_launcher2.add_region_requirement(RegionRequirement(root_locate_region,WRITE_DISCARD,EXCLUSIVE,root_locate_region));
    // compress_launcher2.add_field(0, FID_X);
    // compress_launcher2.add_field(1,FID_X);
    // runtime->execute_task(ctx, compress_launcher2);
    // cout<<"Launching Print After Compress for 2nd Tree"<<endl;
    // runtime->execute_task(ctx, print_launcher2);

    cout<<"Launching Inner Product Task"<<endl;
    InnerProductArgs args(0, 0, overall_max_depth, 0, end_idx, partition_color1, partition_color2, actual_left_depth, tile_height);
    TaskLauncher product_launcher(INNER_PRODUCT_INTER_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    product_launcher.add_region_requirement(RegionRequirement(lr1, READ_ONLY, EXCLUSIVE, lr1));
    product_launcher.add_region_requirement(RegionRequirement(lr2, READ_ONLY, EXCLUSIVE, lr2) );
    product_launcher.add_field(0,FID_X);
    product_launcher.add_field(1,FID_X);
    Future result = runtime->execute_task( ctx, product_launcher );
    cout<<result.get_result<int>()<<endl;

    // Rect<1> gaxpy_tree(0LL, static_cast<coord_t>(pow(2, overall_max_depth )));
    // IndexSpace isgaxpy = runtime->create_index_space(ctx, gaxpy_tree);
    // FieldSpace fsgaxpy = runtime->create_field_space(ctx);
    // {
    //     FieldAllocator allocator = runtime->create_field_allocator(ctx, fsgaxpy);
    //     allocator.allocate_field(sizeof(TreeArgs), FID_X);
    // }
    // LogicalRegion lrgaxpy = runtime->create_logical_region(ctx, isgaxpy, fsgaxpy);
    // Color partition_color3 = 30;
    // GaxpyArgs args(0, 0, 0, overall_max_depth, 0, end_idx, partition_color1, partition_color2, partition_color3, 0, false, false, actual_left_depth, tile_height);
    

    // cout<<"Launching Gaxpy Taks for Tree"<<endl;
    // TaskLauncher gaxpy_launcher(GAXPY_INTER_TASK_ID, TaskArgument(&args, sizeof(GaxpyArgs)));
    // RegionRequirement req1(lr1, READ_ONLY, EXCLUSIVE, lr1);
    // req1.add_field(FID_X);
    // RegionRequirement req2(lr2, READ_ONLY, EXCLUSIVE , lr2);
    // req2.add_field(FID_X);
    // RegionRequirement reqgaxpy(lrgaxpy, WRITE_DISCARD, EXCLUSIVE, lrgaxpy);
    // reqgaxpy.add_field(FID_X);
    // gaxpy_launcher.add_region_requirement(req1);
    // gaxpy_launcher.add_region_requirement(req2);
    // gaxpy_launcher.add_region_requirement(reqgaxpy);
    // runtime->execute_task(ctx, gaxpy_launcher);

    // cout<<"Launching Print Task for Gaxpy"<<endl;
    // TaskLauncher print_gaxpy(PRINT_TASK_ID, TaskArgument(&args2, sizeof(Arguments)));
    // RegionRequirement gaxpy_req( lrgaxpy , READ_ONLY, EXCLUSIVE, lrgaxpy );
    // gaxpy_req.add_field(FID_X);
    // print_gaxpy.add_region_requirement( gaxpy_req );
    // runtime->execute_task(ctx, print_gaxpy );
}


void print_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctxt, HighLevelRuntime *runtime) {
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > read_acc(regions[0], FID_X);
    int node_counter=0;
    int max_depth = args.max_depth;
    queue<Arguments>tree;
    tree.push(args);
    int tile_height = args.tile_height;
    while( !tree.empty() ){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = temp.idx + l - 1 + (1<<(n%tile_height));
        node_counter++;
        cout<<node_counter<<": "<<n<<"~"<<read_acc[idx].lval<<"~"<<idx<<"~"<<read_acc[idx].value<<endl;
        if(!read_acc[idx].is_leaf){
            if((n%tile_height)==(tile_height-1)){
                int tile_nodes = (1<<tile_height)-1;
                coord_t sub_tree_size = (1<<(args.max_depth-n-1))-1;
                coord_t start_idx = temp.idx+tile_nodes;
                coord_t left_level = 2*l;
                coord_t right_level = left_level+1;
                coord_t idx_left_sub_tree = start_idx+left_level*sub_tree_size;
                coord_t idx_right_sub_tree = start_idx+right_level*sub_tree_size;
                Arguments for_left_sub_tree (n+1, 0, 0, max_depth, idx_left_sub_tree, idx_left_sub_tree, temp.partition_color, temp.actual_max_depth, temp.tile_height);
                Arguments for_right_sub_tree(n+1, 0, 0, max_depth, idx_right_sub_tree, idx_right_sub_tree, temp.partition_color, temp.actual_max_depth, temp.tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
            else{
                Arguments for_left_sub_tree (n+1, l * 2    , 0, max_depth, temp.idx, temp.idx, temp.partition_color, temp.actual_max_depth, temp.tile_height);
                Arguments for_right_sub_tree(n+1, l * 2 + 1, 0, max_depth, temp.idx, temp.idx ,temp.partition_color, temp.actual_max_depth, temp.tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
        }
    }
}

void refine_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;
    LogicalRegion my_sub_tree_lr = lr;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(regions[1], FID_X);
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree_acc(regions[0], FID_X);
    coord_t start_idx = args.idx;
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        int actual_l = temp.actual_l;
        coord_t idx = start_idx + l + (1<<(n%tile_height))-1;
        long int node_value=rand();
        node_value = node_value % 10 + 1;
        if (node_value <= 3 || n == max_depth - 1) {
            tree_acc[idx].value = node_value % 3 + 1;
            tree_acc[idx].is_leaf =true;
            tree_acc[idx].lval = actual_l;
        }
        else {
            tree_acc[idx].value = 0;
            tree_acc[idx].lval = actual_l;
        }
        if( (node_value > 3 )&&( n +1 < max_depth ) ){
            if( (n % tile_height )==( tile_height-1 ) ){
                helper_acc[helper_counter].level = l;
                helper_acc[helper_counter].idx = idx;
                helper_acc[helper_counter].n = n;
                helper_acc[helper_counter].launch = true;
                helper_acc[helper_counter].actual_l = actual_l;
                helper_counter++;
            }
            else{
                Arguments for_left_sub_tree (n+1, l * 2    ,2*actual_l, max_depth, temp.idx, 0,temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n+1, l * 2 + 1, 2*actual_l+1 ,max_depth, temp.idx, 0, temp.partition_color, temp.actual_max_depth, tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
        }
    }
}

void compress_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > read_acc(regions[0], FID_X);
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > write_acc(regions[1], FID_X);
    coord_t start_idx = args.idx;
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        int actual_l = temp.actual_l;
        coord_t idx = start_idx + l + (1<<(n%tile_height))-1;
        if(read_acc[idx].is_leaf)
            continue;
        write_acc[helper_counter].actual_l = actual_l;
        write_acc[helper_counter].level = l;
        write_acc[helper_counter].idx = idx;
        write_acc[helper_counter].n = n;
        write_acc[helper_counter].is_valid_entry = true;
        write_acc[helper_counter].launch=false;
        if( ((n % tile_height ) ==( tile_height-1 )) ){
            write_acc[helper_counter].launch = true;
        }
        else if( !read_acc[idx].is_leaf ){
                Arguments for_left_sub_tree (n + 1, l*2   , 2*actual_l, max_depth, temp.idx,0, temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n + 1, l * 2 + 1, 2*actual_l+1, max_depth, temp.idx,0, temp.partition_color, temp.actual_max_depth, tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
        }
        helper_counter++;
    }
}

void compress_update_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > write_acc(regions[0], FID_X);
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(regions[1], FID_X);
    const FieldAccessor<READ_ONLY,RootPosArgs,1,coord_t,Realm::AffineAccessor<RootPosArgs,1,coord_t> > read_child(regions[2], FID_X);
    const FieldAccessor<WRITE_DISCARD,RootPosArgs,1,coord_t,Realm::AffineAccessor<RootPosArgs,1,coord_t> > write_value(regions[3], FID_X);
    int task_counter = args.actual_max_depth;
    int tile_height = args.tile_height;
    for( int i = (1<<tile_height)-1; i>=0 ; i-- ){
        if( !read_acc[i].is_valid_entry )
            continue;
        coord_t idx = read_acc[i].idx;
        if( write_acc[idx].is_leaf )
           continue;
        int nx = read_acc[i].n;
        int l = read_acc[i].level;
        int left_level = 2*l;
        int right_level = 2*l+1;
        coord_t idx_left_sub_tree,idx_right_sub_tree;
        if((nx%tile_height)==(tile_height-1)){
                int rightChildVal = read_child[task_counter--].value;
                int leftChildVal = read_child[task_counter--].value;
                write_acc[idx].value = leftChildVal + rightChildVal;
        }
        else{
                idx_left_sub_tree = args.idx + left_level + (1<<((nx+1)%tile_height))-1;
                idx_right_sub_tree = args.idx + right_level + (1<<((nx+1)%tile_height))-1;
                write_acc[idx].value = write_acc[idx_left_sub_tree].value + write_acc[idx_right_sub_tree].value;
        }
    }
    write_value[args.root_location].value = write_acc[args.idx].value;
}




void reconstruct_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;
    LogicalRegion my_sub_tree_lr = lr;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(regions[1], FID_X);
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree_acc(regions[0], FID_X);
    coord_t start_idx = args.idx;
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        int actual_l = temp.actual_l;
        int carry = temp.carry;
        coord_t idx = start_idx + l + (1<<(n%tile_height))-1;
        if(tree_acc[idx].is_leaf){
            tree_acc[idx].value+=carry;
            continue;
        }
        else{
            int val = tree_acc[idx].value+carry;
            val/=2;
            tree_acc[idx].value=0;
            if( (n % tile_height )==( tile_height-1 ) ){
                helper_acc[helper_counter].level = l;
                helper_acc[helper_counter].idx = idx;
                helper_acc[helper_counter].n = n;
                helper_acc[helper_counter].launch = true;
                helper_acc[helper_counter].actual_l = actual_l;
                helper_acc[helper_counter].carry=val;
                helper_counter++;
            }
            else{
                Arguments for_left_sub_tree (n+1, l * 2    ,2*actual_l, max_depth, temp.idx, 0,temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n+1, l * 2 + 1, 2*actual_l+1 ,max_depth, temp.idx, 0, temp.partition_color, temp.actual_max_depth, tile_height);
                for_left_sub_tree.carry=val;
                for_right_sub_tree.carry=val;
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
        }
    }
}

int norm_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    queue<Arguments>tree;
    tree.push(args);
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = LogicalPartition::NO_PART;
    LogicalRegion my_sub_tree_lr = lr;
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(regions[1], FID_X);
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree_acc(regions[0], FID_X);
    coord_t start_idx = args.idx;
    int result=0;
    while(!tree.empty()){
        Arguments temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        int actual_l = temp.actual_l;
        coord_t idx = start_idx + l + (1<<(n%tile_height))-1;
        result+=tree_acc[idx].value*tree_acc[idx].value;
        if(tree_acc[idx].is_leaf){
            continue;
        }
        else{
            if( (n % tile_height )==( tile_height-1 ) ){
                helper_acc[helper_counter].level = l;
                helper_acc[helper_counter].idx = idx;
                helper_acc[helper_counter].n = n;
                helper_acc[helper_counter].launch = true;
                helper_acc[helper_counter].actual_l = actual_l;
                helper_counter++;
            }
            else{
                Arguments for_left_sub_tree (n+1, l * 2    ,2*actual_l, max_depth, temp.idx, 0,temp.partition_color, temp.actual_max_depth, tile_height);
                Arguments for_right_sub_tree(n+1, l * 2 + 1, 2*actual_l+1 ,max_depth, temp.idx, 0, temp.partition_color, temp.actual_max_depth, tile_height);
                tree.push( for_left_sub_tree );
                tree.push( for_right_sub_tree );
            }
        }
    }
    return result;
}


int inner_product_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    InnerProductArgs args = task->is_index_space ? *(const InnerProductArgs *) task->local_args
    : *(const InnerProductArgs *) task->args;
    queue<InnerProductArgs>tree;
    tree.push(args);
    int max_depth = args.max_depth;
    int tile_height = args.tile_height;
    int helper_counter=0;
    const FieldAccessor<WRITE_DISCARD,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > helper_acc(regions[2], FID_X);
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree1(regions[0], FID_X);
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree2(regions[1], FID_X);
    coord_t start_idx = args.idx;
    int result=0;
    while(!tree.empty()){
        InnerProductArgs temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        coord_t idx = start_idx + l + (1<<(n%tile_height))-1;
        bool leaf1 = tree1[idx].is_leaf;
        bool leaf2 = tree2[idx].is_leaf;
        result = result + tree1[idx].value*tree2[idx].value;
        if(leaf1||leaf2)
            continue;
        if((n% tile_height )==( tile_height-1 )){
            helper_acc[helper_counter].n = n;
            helper_acc[helper_counter].level =l;
            helper_acc[helper_counter].idx = idx;
            helper_acc[helper_counter].launch=true;
            helper_counter++;
        }
        else{
            InnerProductArgs for_left_sub_tree (n + 1, l * 2    , max_depth, temp.idx, temp.end_idx, temp.partition_color1, temp.partition_color2, temp.actual_max_depth, tile_height);
            InnerProductArgs for_right_sub_tree(n + 1, l * 2 + 1, max_depth, temp.idx, temp.end_idx, temp.partition_color1, temp.partition_color2 ,temp.actual_max_depth, tile_height);
            tree.push( for_left_sub_tree );
            tree.push( for_right_sub_tree );
        }
    }
    return result;
}


void gaxpy_intra_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    GaxpyArgs args = task->is_index_space ? *(const GaxpyArgs *) task->local_args
    : *(const GaxpyArgs *) task->args;
    int tile_height = args.tile_height;
    queue<GaxpyArgs>tree;
    tree.push(args);
    int helper_counter=0;
    int max_depth = args.max_depth;
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree1(regions[0], FID_X);
    const FieldAccessor<READ_ONLY,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree2(regions[1], FID_X);
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree3(regions[2], FID_X);    
    const FieldAccessor<WRITE_DISCARD,GaxpyHelper,1,coord_t,Realm::AffineAccessor<GaxpyHelper,1,coord_t> > helper_acc(regions[3], FID_X); 
    coord_t start_idx = args.idx;
    while(!tree.empty()){
        GaxpyArgs temp = tree.front();
        tree.pop();
        int n = temp.n;
        int l = temp.l;
        int pass = temp.pass;
        int actual_l = temp.actual_l;
        coord_t idx = start_idx + l + (1<<(n%tile_height))-1;
        bool left_null = temp.left_null;
        bool right_null = temp.right_null;
        int value;
        if( n > max_depth )
            break;
        tree3[idx].lval = actual_l;
        if( left_null ){
            if(tree2[idx].is_leaf){
                value = pass + tree2[idx].value;
                tree3[idx].value = value;
                tree3[idx].is_leaf = true;
            }
            else{
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = pass/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].actual_l = actual_l;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, 2*actual_l, max_depth, temp.idx,0, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2+1, 2*actual_l+1, max_depth, temp.idx,0, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }
            }
        }
        else if( right_null ){
            if( tree1[idx].is_leaf){
                value = pass + tree1[idx].value;
                tree3[idx].value = value;
                tree3[idx].is_leaf = true;
            }
            else{
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = pass/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].actual_l = actual_l;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, 2*actual_l,max_depth, temp.idx,0, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2+1, 2*actual_l+1 ,max_depth, temp.idx,0, temp.partition_color1, temp.partition_color2, temp.partition_color3, pass/2, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
        }
        else{
            if( (tree1[idx].is_leaf )&&( tree2[idx].is_leaf )){
                value = tree1[idx].value + tree2[idx].value;
                tree3[idx].value = value;
                tree3[idx].is_leaf = true;
            }
            else if(tree1[idx].is_leaf){
                value = tree1[idx].value;
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = value/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].actual_l = actual_l;
                    helper_acc[helper_counter].left_null = true;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, 2*actual_l,max_depth, temp.idx,0, temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, true, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2+1, 2*actual_l+1,max_depth, temp.idx,0, temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, true, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
            else if(tree2[idx].is_leaf){
                value = tree2[idx].value;
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = value/2;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].actual_l = actual_l;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = true;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, 2*actual_l,max_depth, temp.idx, 0,temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, left_null, true, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2+1, 2*actual_l+1,max_depth, temp.idx,0, temp.partition_color1, temp.partition_color2, temp.partition_color3, value/2, left_null, true, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
            else{
                if((n%tile_height)==(tile_height-1)){
                    helper_acc[helper_counter].n=n;
                    helper_acc[helper_counter].l=l;
                    helper_acc[helper_counter].pass = 0;
                    helper_acc[helper_counter].launch = true;
                    helper_acc[helper_counter].idx = idx;
                    helper_acc[helper_counter].actual_l = actual_l;
                    helper_acc[helper_counter].left_null = left_null;
                    helper_acc[helper_counter].right_null = right_null;
                    helper_counter++;
                }
                else{
                    GaxpyArgs for_left_sub_tree( n+1, l*2, 2*actual_l,max_depth, temp.idx, 0, temp.partition_color1, temp.partition_color2, temp.partition_color3, 0, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    GaxpyArgs for_right_sub_tree(n+1, l*2+1, 2*actual_l+1,max_depth, temp.idx, 0, temp.partition_color1, temp.partition_color2, temp.partition_color3, 0, left_null, right_null, temp.actual_max_depth, temp.tile_height);
                    tree.push( for_left_sub_tree );
                    tree.push( for_right_sub_tree );
                }   
            }
        }
    }
}

void gaxpy_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    GaxpyArgs args = task->is_index_space ? *(const GaxpyArgs *) task->local_args
    : *(const GaxpyArgs *) task->args;
    int tile_height = args.tile_height;
    tile_height = min(tile_height,args.max_depth-args.n);
    int tile_nodes = (1<<tile_height)-1;
    int n = args.n;
    coord_t idx = args.idx;
    LogicalRegion lr1,subtree1,childtree1,lr2,subtree2,childtree2,lr;
    LogicalPartition lp1,lp2;
    if(!args.left_null){
        lr1 = regions[0].get_logical_region();
        lp1 = runtime->get_logical_partition_by_color(ctx,lr1,args.partition_color1);
        subtree1 = runtime->get_logical_subregion_by_color(ctx, lp1, 0);
        if(idx+tile_nodes < args.end_idx)
            childtree1 = runtime->get_logical_subregion_by_color(ctx,lp1,1);
    }
    if(!args.right_null){
        if(args.left_null)
            lr2 = regions[0].get_logical_region();
        else
            lr2 = regions[1].get_logical_region();
        lp2 = runtime->get_logical_partition_by_color(ctx,lr2,args.partition_color2);
        subtree2 = runtime->get_logical_subregion_by_color(ctx, lp2, 0);
        if(idx+tile_nodes < args.end_idx)
            childtree2 = runtime->get_logical_subregion_by_color(ctx,lp2,1);
    }
    if((args.left_null)||(args.right_null))
        lr = regions[1].get_logical_region();
    else
        lr = regions[2].get_logical_region();
    DomainPointColoring colorStartTile;
    LogicalRegion subtree = lr;
    LogicalRegion childtree;
    LogicalPartition lp;
    IndexSpace is;
    IndexPartition ip;
    if(idx+tile_nodes < args.end_idx ){
        colorStartTile[0] = Rect<1>(idx,idx+tile_nodes-1);
        colorStartTile[1] = Rect<1>(idx+tile_nodes,args.end_idx);
        Rect<1>color_space = Rect<1>(0,1);
        is = lr.get_index_space();
        ip = runtime->create_index_partition(ctx, is, color_space, colorStartTile, DISJOINT_KIND, args.partition_color3);
        lp = runtime->get_logical_partition(ctx, lr, ip);
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
        childtree = runtime->get_logical_subregion_by_color(ctx,lp,1);
    }
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    Rect<1> dummy_Array(0,0);
    is = runtime->create_index_space(ctx, dummy_Array);
    fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(TreeArgs), FID_X);
    }
    LogicalRegion dummy_region = runtime->create_logical_region(ctx, is, fs);
    RegionRequirement reqd(dummy_region, WRITE_DISCARD, EXCLUSIVE , dummy_region);
    RegionRequirement req4(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req4.add_field(FID_X);
    reqd.add_field(FID_X);
    if( args.left_null ){
        RegionRequirement req2(subtree2, READ_ONLY, EXCLUSIVE, lr2);
        req2.add_field(FID_X);
        RegionRequirement req3(subtree, WRITE_DISCARD, EXCLUSIVE, lr);
        req3.add_field(FID_X);
        TaskLauncher gaxpy_intra_launcher(GAXPY_INTRA_TASK_ID, TaskArgument(&args,sizeof(GaxpyArgs)));
        gaxpy_intra_launcher.add_region_requirement(reqd);
        gaxpy_intra_launcher.add_region_requirement(req2);
        gaxpy_intra_launcher.add_region_requirement(req3);
        gaxpy_intra_launcher.add_region_requirement(req4);
        runtime->execute_task(ctx,gaxpy_intra_launcher);
    }
    else if(args.right_null){
        RegionRequirement req1(subtree1, READ_ONLY, EXCLUSIVE, lr1);
        req1.add_field(FID_X);
        RegionRequirement req3(subtree, WRITE_DISCARD, EXCLUSIVE, lr);
        req3.add_field(FID_X);
        TaskLauncher gaxpy_intra_launcher(GAXPY_INTRA_TASK_ID, TaskArgument(&args,sizeof(GaxpyArgs)));
        gaxpy_intra_launcher.add_region_requirement(req1);
        gaxpy_intra_launcher.add_region_requirement(reqd);
        gaxpy_intra_launcher.add_region_requirement(req3);
        gaxpy_intra_launcher.add_region_requirement(req4);
        runtime->execute_task(ctx,gaxpy_intra_launcher);
    }
    else{
        RegionRequirement req1(subtree1, READ_ONLY, EXCLUSIVE, lr1);
        req1.add_field(FID_X);
        RegionRequirement req2(subtree2, READ_ONLY, EXCLUSIVE, lr2);
        req2.add_field(FID_X);
        RegionRequirement req3(subtree, WRITE_DISCARD, EXCLUSIVE, lr);
        req3.add_field(FID_X);
        TaskLauncher gaxpy_intra_launcher(GAXPY_INTRA_TASK_ID, TaskArgument(&args,sizeof(GaxpyArgs)));
        gaxpy_intra_launcher.add_region_requirement(req1);
        gaxpy_intra_launcher.add_region_requirement(req2);
        gaxpy_intra_launcher.add_region_requirement(req3);
        gaxpy_intra_launcher.add_region_requirement(req4);
        runtime->execute_task(ctx,gaxpy_intra_launcher);
    }
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req4 );
    const FieldAccessor<READ_ONLY,GaxpyHelper,1,coord_t,Realm::AffineAccessor<GaxpyHelper,1,coord_t> > read_acc(physicalRegion, FID_X);
    vector<pair<coord_t,coord_t> >color_index_left, color_index_right, color_index_both;
    coord_t sub_tree_size = (1<<(args.max_depth-n-tile_height))-1;
    coord_t start_idx = args.idx+tile_nodes;
    vector<pair<coord_t,coord_t> >color_index;
    vector<GaxpyArgs>argsReqd;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++){
        if(!read_acc[i].launch)
            break;
        int nx = read_acc[i].n;
        int pass = read_acc[i].pass;
        int actual_l = read_acc[i].actual_l;
        int level = read_acc[i].l;
        coord_t left_level = 2*level;
        coord_t right_level = left_level+1;
        bool left_null = read_acc[i].left_null;
        bool right_null = read_acc[i].right_null;
        coord_t idx_left_sub_tree = start_idx+left_level*sub_tree_size;
        coord_t idx_right_sub_tree = start_idx+right_level*sub_tree_size;
        GaxpyArgs left_args( nx+1, 0, 2*actual_l, args.max_depth, idx_left_sub_tree, idx_right_sub_tree-1, args.partition_color1, args.partition_color2, args.partition_color3, pass, left_null, right_null , args.actual_max_depth, args.tile_height);
        GaxpyArgs right_args( nx+1, 0 , 2*actual_l+1 ,args.max_depth, idx_right_sub_tree,idx_right_sub_tree + sub_tree_size-1 ,args.partition_color1, args.partition_color2, args.partition_color3, pass, left_null, right_null , args.actual_max_depth, args.tile_height);
        argsReqd.push_back(left_args);
        argsReqd.push_back(right_args);
        color_index.push_back(make_pair(idx_left_sub_tree,idx_right_sub_tree-1));
        color_index.push_back(make_pair(idx_right_sub_tree, idx_right_sub_tree +  sub_tree_size-1 ) );
    }
    if(color_index.size() > 0 ){
        is = childtree.get_index_space();
        DomainPointColoring coloring;
        for( int i = 0 ; i < color_index.size() ; i++ ){
            coloring[i]= Rect<1>(color_index[i].first,color_index[i].second);
        }
        Rect<1>color_space = Rect<1>(0,color_index.size()-1);
        ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color3);
        lp = runtime->get_logical_partition(ctx, childtree, ip);
    }
    for( int i = 0 ; i < argsReqd.size(); i++ ){
        GaxpyArgs currentArg = argsReqd[i];
        TaskLauncher gaxpy_launcher(GAXPY_INTER_TASK_ID,TaskArgument(&currentArg,sizeof(GaxpyArgs)));
        LogicalRegion currentTile = runtime->get_logical_subregion_by_color(ctx,lp,i);
        if(currentArg.left_null){
            gaxpy_launcher.add_region_requirement(RegionRequirement(childtree2,READ_ONLY,EXCLUSIVE,lr2));
            gaxpy_launcher.add_region_requirement(RegionRequirement(currentTile,WRITE_DISCARD,EXCLUSIVE,lr));
            gaxpy_launcher.add_field(0,FID_X);
            gaxpy_launcher.add_field(1,FID_X);
            runtime->execute_task(ctx,gaxpy_launcher);
        }
        else if(currentArg.right_null){
            gaxpy_launcher.add_region_requirement(RegionRequirement(childtree1,READ_ONLY,EXCLUSIVE,lr1));
            gaxpy_launcher.add_region_requirement(RegionRequirement(currentTile,WRITE_DISCARD,EXCLUSIVE,lr));
            gaxpy_launcher.add_field(0,FID_X);
            gaxpy_launcher.add_field(1,FID_X);
            runtime->execute_task(ctx,gaxpy_launcher);
        }
        else{
            gaxpy_launcher.add_region_requirement(RegionRequirement(childtree1,READ_ONLY,EXCLUSIVE,lr1));
            gaxpy_launcher.add_region_requirement(RegionRequirement(childtree2,READ_ONLY,EXCLUSIVE,lr2));
            gaxpy_launcher.add_region_requirement(RegionRequirement(currentTile,WRITE_DISCARD,EXCLUSIVE,lr));
            gaxpy_launcher.add_field(0,FID_X);
            gaxpy_launcher.add_field(1,FID_X);
            gaxpy_launcher.add_field(2,FID_X);
            runtime->execute_task(ctx,gaxpy_launcher);
        }
    }
}

int inner_product_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    InnerProductArgs args = task->is_index_space ? *(const InnerProductArgs *) task->local_args
    : *(const InnerProductArgs *) task->args;
    int tile_height = args.tile_height;
    tile_height = min(tile_height,args.max_depth-args.n);
    int tile_nodes = (1<<tile_height)-1;
    int n = args.n;
    LogicalRegion subtree1,childtree1,subtree2,childtree2;
    LogicalRegion lr1 = regions[0].get_logical_region();
    LogicalRegion lr2 = regions[1].get_logical_region();
    LogicalPartition lp1 = runtime->get_logical_partition_by_color(ctx,lr1,args.partition_color1);
    LogicalPartition lp2 = runtime->get_logical_partition_by_color(ctx,lr2,args.partition_color2);

    if(args.idx + tile_nodes < args.end_idx ){
        subtree1 = runtime->get_logical_subregion_by_color(ctx, lp1, 0);
        childtree1 = runtime->get_logical_subregion_by_color(ctx,lp1,1);
        subtree2 = runtime->get_logical_subregion_by_color(ctx, lp2, 0);
        childtree2 = runtime->get_logical_subregion_by_color(ctx,lp2,1);
    }
    else{
        subtree1 = runtime->get_logical_subregion_by_color(ctx, lp1, 0);
        subtree2 = runtime->get_logical_subregion_by_color(ctx, lp2, 0);
    }
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher inner_product_intra_launcher(INNER_PRODUCT_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments) ) );
    RegionRequirement req1(subtree1, READ_ONLY, EXCLUSIVE, lr1);
    RegionRequirement req2(subtree2,READ_ONLY,EXCLUSIVE,lr2);
    RegionRequirement req3(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    req3.add_field(FID_X);
    inner_product_intra_launcher.add_region_requirement(req1);
    inner_product_intra_launcher.add_region_requirement(req2);
    inner_product_intra_launcher.add_region_requirement(req3);
    Future tile_result = runtime->execute_task(ctx,inner_product_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req3 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    coord_t sub_tree_size = (1<<(args.max_depth-n-tile_height))-1;
    coord_t start_idx = args.idx+tile_nodes;
    int task_counter=0;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
        if(!read_acc[i].launch){
            break;
        }
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        coord_t left_level = 2*level;
        coord_t right_level = left_level+1;
        coord_t idx_left_sub_tree = start_idx+left_level*sub_tree_size;
        coord_t idx_right_sub_tree = start_idx+right_level*sub_tree_size;
        InnerProductArgs left_args( nx+1 , 0, args.max_depth, idx_left_sub_tree , idx_right_sub_tree-1, args.partition_color1 , args.partition_color2, args.actual_max_depth , args.tile_height);
        InnerProductArgs right_args( nx+1 , 0, args.max_depth, idx_right_sub_tree , idx_right_sub_tree + sub_tree_size-1 , args.partition_color1, args.partition_color2 ,args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
    }
    int result=0;
    if( task_counter > 0 ){
        lp1 = runtime->get_logical_partition_by_color(ctx,childtree1,args.partition_color1);
        lp2 = runtime->get_logical_partition_by_color(ctx,childtree2,args.partition_color2);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher product_launcher(INNER_PRODUCT_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        product_launcher.add_region_requirement(RegionRequirement(lp1,0,READ_ONLY, EXCLUSIVE, lr1));
        product_launcher.add_region_requirement(RegionRequirement(lp2,0,READ_ONLY, EXCLUSIVE, lr2));
        product_launcher.add_field(0,FID_X);
        product_launcher.add_field(1,FID_X);
        FutureMap f_result = runtime->execute_index_space(ctx, product_launcher);
        for( int i = 0 ; i < task_counter ; i++ )
            result = result + f_result.get_result<int>(i);
    }
    result+=tile_result.get_result<int>();
    return result;
}

int norm_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    tile_height = min(tile_height,args.max_depth-args.n);
    int tile_nodes = (1<<tile_height)-1;
    int n = args.n;
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = runtime->get_logical_partition_by_color(ctx,lr,args.partition_color);
    LogicalRegion subtree,childtree;
    if( args.idx + tile_nodes < args.end_idx ){
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
        childtree = runtime->get_logical_subregion_by_color(ctx,lp,1);
    }
    else{
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    }
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher norm_intra_launcher(NORM_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments) ) );
    RegionRequirement req1(subtree, READ_ONLY, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    norm_intra_launcher.add_region_requirement(req1);
    norm_intra_launcher.add_region_requirement(req2);
    Future tile_result = runtime->execute_task(ctx,norm_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    coord_t sub_tree_size = (1<<(args.max_depth-n-tile_height))-1;
    coord_t start_idx = args.idx+tile_nodes;
    int task_counter=0;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
        if(!read_acc[i].launch){
            break;
        }
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        int actual_l = read_acc[i].actual_l;
        coord_t left_level = 2*level;
        coord_t right_level = left_level+1;
        coord_t idx_left_sub_tree = start_idx+left_level*sub_tree_size;
        coord_t idx_right_sub_tree = start_idx+right_level*sub_tree_size;
        Arguments left_args( nx+1,0 ,2*actual_l ,args.max_depth, idx_left_sub_tree , idx_right_sub_tree-1 ,args.partition_color , args.actual_max_depth, args.tile_height);
        Arguments right_args( nx+1,0, 2*actual_l+1, args.max_depth, idx_right_sub_tree , idx_right_sub_tree + sub_tree_size-1  ,args.partition_color, args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
    }
    FutureMap child_result;
    if( task_counter > 0 ){
        lp = runtime->get_logical_partition_by_color(ctx,childtree,args.partition_color);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher norm_launcher(NORM_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        norm_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        norm_launcher.add_field(0, FID_X);
        child_result =runtime->execute_index_space(ctx, norm_launcher);
    }
    int result=tile_result.get_result<int>();
    for( int i = 0 ; i < task_counter; i++ )
        result+=child_result.get_result<int>(i);
    return result;
}


void reconstruct_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    tile_height = min(tile_height,args.max_depth-args.n);
    int tile_nodes = (1<<tile_height)-1;
    int n = args.n;
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalPartition lp = runtime->get_logical_partition_by_color(ctx,lr,args.partition_color);
    LogicalRegion subtree,childtree;
    if(args.idx + tile_nodes < args.end_idx ){
            subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
            childtree = runtime->get_logical_subregion_by_color(ctx,lp,1);
    }
    else{
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    }
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher reconstruct_intra_launcher(RECONSTRUCT_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments) ) );
    RegionRequirement req1(subtree, WRITE_DISCARD, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    reconstruct_intra_launcher.add_region_requirement(req1);
    reconstruct_intra_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,reconstruct_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    coord_t sub_tree_size = (1<<(args.max_depth-n-tile_height))-1;
    coord_t start_idx = args.idx+tile_nodes;
    int task_counter=0;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
        if(!read_acc[i].launch){
            break;
        }
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        int actual_l = read_acc[i].actual_l;
        int carry = read_acc[i].carry;
        coord_t left_level = 2*level;
        coord_t right_level = left_level+1;
        coord_t idx_left_sub_tree = start_idx+left_level*sub_tree_size;
        coord_t idx_right_sub_tree = start_idx+right_level*sub_tree_size;
        Arguments left_args( nx+1,0 ,2*actual_l ,args.max_depth, idx_left_sub_tree , idx_right_sub_tree-1 ,args.partition_color , args.actual_max_depth, args.tile_height);
        left_args.carry=carry;
        Arguments right_args( nx+1,0, 2*actual_l+1, args.max_depth, idx_right_sub_tree , idx_right_sub_tree + sub_tree_size-1  ,args.partition_color, args.actual_max_depth, args.tile_height);
        right_args.carry=carry;
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
    }
    if( task_counter > 0 ){
        lp = runtime->get_logical_partition_by_color(ctx,childtree,args.partition_color);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher reconstruct_launcher(RECONSTRUCT_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        reconstruct_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        reconstruct_launcher.add_field(0, FID_X);
        runtime->execute_index_space(ctx, reconstruct_launcher);
    }
}


void compress_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    tile_height = min(tile_height,args.max_depth-args.n);
    int tile_nodes = (1<<tile_height)-1;
    int n = args.n;
    LogicalRegion lr = regions[0].get_logical_region();
    LogicalRegion root_locate, root_locate_region;
    LogicalPartition lp = runtime->get_logical_partition_by_color(ctx,lr,args.partition_color);
    LogicalRegion subtree,childtree;
    if(args.idx+tile_nodes < args.end_idx ){
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
        childtree = runtime->get_logical_subregion_by_color(ctx,lp,1);
    }
    else{
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);   
    }
    root_locate = regions[1].get_logical_region();
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height)));
    IndexSpace is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher compress_intra_launcher(COMPRESS_INTRA_TASK_ID, TaskArgument(&args,sizeof(Arguments)));
    RegionRequirement req1(subtree, WRITE_DISCARD, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    compress_intra_launcher.add_region_requirement(req1);
    compress_intra_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,compress_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    int task_counter=0;
    coord_t sub_tree_size = (1<<(args.max_depth-n-tile_height))-1;
    coord_t start_idx = args.idx+tile_nodes;
    for( int  i = 0 ; i < (1<<tile_height) ; i++){
        if(!read_acc[i].is_valid_entry)
            continue;
        bool launch = read_acc[i].launch;
        if( launch ){
            int level = read_acc[i].level;
            int nx = read_acc[i].n;
            int actual_l = read_acc[i].actual_l;
            coord_t left_level = 2*level;
            coord_t right_level = left_level+1;
            coord_t idx_left_sub_tree = start_idx+left_level*sub_tree_size;
            coord_t idx_right_sub_tree = start_idx+right_level*sub_tree_size;
            Arguments left_args( nx+1,0 ,2*actual_l ,args.max_depth, idx_left_sub_tree , idx_right_sub_tree-1 ,args.partition_color , args.actual_max_depth , args.tile_height,task_counter);
            arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
            task_counter++;
            Arguments right_args( nx+1,0, 2*actual_l+1, args.max_depth, idx_right_sub_tree , idx_right_sub_tree + sub_tree_size-1  ,args.partition_color, args.actual_max_depth, args.tile_height,task_counter);
            arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
            task_counter++;
        }
    }

    if( task_counter > 0 ){
        Rect<1> root_location(0, task_counter-1);
        is = runtime->create_index_space(ctx, root_location);
        fs = runtime->create_field_space(ctx);
        {
            FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
            allocator.allocate_field(sizeof(RootPosArgs), FID_X);
        }
        root_locate_region = runtime->create_logical_region(ctx, is, fs);
        lp = runtime->get_logical_partition_by_color(ctx,childtree,args.partition_color);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher compress_launcher(COMPRESS_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        compress_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        IndexSpace is2 = root_locate_region.get_index_space();
        DomainPointColoring coloring;
        for( int i = 0 ; i < task_counter ; i++ ){
            coloring[i]= Rect<1>(i,i);
        }
        Rect<1>color_space = Rect<1>(0,task_counter-1);
        IndexPartition ip2 = runtime->create_index_partition(ctx, is2, color_space, coloring, DISJOINT_KIND, args.partition_color);
        LogicalPartition lp2 = runtime->get_logical_partition(ctx, root_locate_region, ip2);
        compress_launcher.add_region_requirement(RegionRequirement(lp2,0,WRITE_DISCARD,EXCLUSIVE,root_locate_region));
        compress_launcher.add_field(0, FID_X);
        compress_launcher.add_field(1,FID_X);
        runtime->execute_index_space(ctx, compress_launcher);
    }
    if(task_counter == 0 ){
            Rect<1> root_location(0, 0);
            is = runtime->create_index_space(ctx, root_location);
            fs = runtime->create_field_space(ctx);
            {
                FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
                allocator.allocate_field(sizeof(RootPosArgs), FID_X);
            }
            root_locate_region = runtime->create_logical_region(ctx, is, fs);
    }
    args.actual_max_depth = task_counter-1;
    TaskLauncher compress_update_launcher(COMPRESS_UPDATE_TASK_ID, TaskArgument(&args, sizeof(Arguments)));
    RegionRequirement req4(subtree,WRITE_DISCARD,EXCLUSIVE,lr);
    RegionRequirement req5(new_helper_Region,READ_ONLY,EXCLUSIVE,new_helper_Region);
    RegionRequirement req6( root_locate_region , WRITE_DISCARD, EXCLUSIVE, root_locate_region );
    RegionRequirement req7( root_locate, WRITE_DISCARD, EXCLUSIVE, root_locate );
    req4.add_field(FID_X);
    req5.add_field(FID_X);
    req6.add_field(FID_X);
    req7.add_field(FID_X);
    compress_update_launcher.add_region_requirement( req4 );
    compress_update_launcher.add_region_requirement( req5 );
    compress_update_launcher.add_region_requirement( req6 );
    compress_update_launcher.add_region_requirement( req7 );
    runtime->execute_task(ctx, compress_update_launcher);

}

void refine_inter_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, HighLevelRuntime *runtime){
    
    Arguments args = task->is_index_space ? *(const Arguments *) task->local_args
    : *(const Arguments *) task->args;
    int tile_height = args.tile_height;
    tile_height = min(tile_height,args.max_depth-args.n);
    int tile_nodes = (1<<tile_height)-1;
    const FieldAccessor<WRITE_DISCARD,TreeArgs,1,coord_t,Realm::AffineAccessor<TreeArgs,1,coord_t> > tree_acc(regions[0], FID_X);
    coord_t idx = args.idx;
    LogicalRegion lr = regions[0].get_logical_region();
    assert(lr != LogicalRegion::NO_REGION);
    DomainPointColoring colorStartTile;
    LogicalRegion subtree = lr;
    LogicalRegion childtree;
    IndexSpace is;
    if(idx+tile_nodes < args.end_idx ){
        colorStartTile[0] = Rect<1>(idx,idx+tile_nodes-1);
        colorStartTile[1] = Rect<1>(idx+tile_nodes,args.end_idx);
        Rect<1>color_space = Rect<1>(0,1);
        is = lr.get_index_space();
        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, colorStartTile, DISJOINT_KIND, args.partition_color);
        LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
        childtree = runtime->get_logical_subregion_by_color(ctx,lp,1);
    }
    else{
        colorStartTile[0] = Rect<1>(idx,idx+tile_nodes-1);
        Rect<1>color_space = Rect<1>(0,0);
        is = lr.get_index_space();
        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, colorStartTile, DISJOINT_KIND, args.partition_color);
        LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);
        subtree = runtime->get_logical_subregion_by_color(ctx, lp, 0);
    }
    Rect<1> helper_Array(0LL, static_cast<coord_t>(pow(2, tile_height-1)));
    is = runtime->create_index_space(ctx, helper_Array);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(HelperArgs), FID_X);
    }
    LogicalRegion new_helper_Region = runtime->create_logical_region(ctx, is, fs);
    TaskLauncher refine_intra_launcher(REFINE_INTRA_TASK_ID, TaskArgument(&args, sizeof(Arguments) ) );
    RegionRequirement req1(subtree, WRITE_DISCARD, EXCLUSIVE, lr);
    RegionRequirement req2(new_helper_Region, WRITE_DISCARD, EXCLUSIVE, new_helper_Region);
    req1.add_field(FID_X);
    req2.add_field(FID_X);
    refine_intra_launcher.add_region_requirement(req1);
    refine_intra_launcher.add_region_requirement(req2);
    runtime->execute_task(ctx,refine_intra_launcher);
    ArgumentMap arg_map;
    PhysicalRegion physicalRegion = runtime->map_region( ctx, req2 );
    const FieldAccessor<READ_ONLY,HelperArgs,1,coord_t,Realm::AffineAccessor<HelperArgs,1,coord_t> > read_acc(physicalRegion, FID_X);
    int task_counter=0;
    vector<pair<coord_t,coord_t> >color_index;
    int n = args.n;
    coord_t sub_tree_size = (1<<(args.max_depth-n-tile_height))-1;
    coord_t start_idx = args.idx+tile_nodes;
    for( int i = 0 ; i < (1<<(tile_height-1)); i++ ){
        if(!read_acc[i].launch){
            break;
        }
        int level = read_acc[i].level;
        int nx = read_acc[i].n;
        int actual_l = read_acc[i].actual_l;
        coord_t left_level = 2*level;
        coord_t right_level = left_level+1;
        coord_t idx_left_sub_tree = start_idx+left_level*sub_tree_size;
        coord_t idx_right_sub_tree = start_idx+right_level*sub_tree_size;
        Arguments left_args( nx+1,0 ,2*actual_l ,args.max_depth, idx_left_sub_tree , idx_right_sub_tree-1 ,args.partition_color , args.actual_max_depth , args.tile_height);
        Arguments right_args( nx+1,0, 2*actual_l+1, args.max_depth, idx_right_sub_tree , idx_right_sub_tree + sub_tree_size-1  ,args.partition_color, args.actual_max_depth, args.tile_height);
        arg_map.set_point( task_counter , TaskArgument(&left_args,sizeof(Arguments)));
        task_counter++;
        arg_map.set_point( task_counter, TaskArgument(&right_args, sizeof(Arguments)));
        task_counter++;
        color_index.push_back(make_pair(idx_left_sub_tree,idx_right_sub_tree-1));
        color_index.push_back(make_pair(idx_right_sub_tree, idx_right_sub_tree +  sub_tree_size-1 ) );
    }
    if( task_counter > 0 ){
        IndexSpace is = childtree.get_index_space();
        DomainPointColoring coloring;
        for( int i = 0 ; i < task_counter ; i++ ){
            coloring[i]= Rect<1>(color_index[i].first,color_index[i].second);
        }
        Rect<1>color_space = Rect<1>(0,task_counter-1);
        IndexPartition ip = runtime->create_index_partition(ctx, is, color_space, coloring, DISJOINT_KIND, args.partition_color);
        LogicalPartition lp = runtime->get_logical_partition(ctx, childtree, ip);
        Rect<1> launch_domain(0,task_counter-1);
        IndexTaskLauncher refine_launcher(REFINE_INTER_TASK_ID, launch_domain, TaskArgument(NULL, 0), arg_map);
        refine_launcher.add_region_requirement(RegionRequirement(lp,0,WRITE_DISCARD, EXCLUSIVE, lr));
        refine_launcher.add_field(0, FID_X);
        runtime->execute_index_space(ctx, refine_launcher);
    }
}

int main(int argc, char** argv){

    srand(time(NULL));
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }

    {
        TaskVariantRegistrar registrar(REFINE_INTER_TASK_ID, "refine_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<refine_inter_task>(registrar, "refine_inter");
    }

    {
        TaskVariantRegistrar registrar(REFINE_INTRA_TASK_ID, "refine_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<refine_intra_task>(registrar, "refine_intra");
    }

    {
        TaskVariantRegistrar registrar(PRINT_TASK_ID, "print");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<print_task>(registrar, "print");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_INTER_TASK_ID, "compress_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<compress_inter_task>(registrar, "compress_inter");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_INTRA_TASK_ID, "compress_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<compress_intra_task>(registrar, "compress_intra");
    }

    {
        TaskVariantRegistrar registrar(COMPRESS_UPDATE_TASK_ID, "compress_update");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<compress_update_task>(registrar, "compress_update");

    }

    {
        TaskVariantRegistrar registrar(RECONSTRUCT_INTER_TASK_ID, "reconstruct_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<reconstruct_inter_task>(registrar, "reconstruct_inter");
    }

    {
        TaskVariantRegistrar registrar(RECONSTRUCT_INTRA_TASK_ID, "reconstruct_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<reconstruct_intra_task>(registrar, "reconstruct_intra");
    }

    {
        TaskVariantRegistrar registrar(NORM_INTER_TASK_ID, "norm_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<int,norm_inter_task>(registrar, "norm_inter");
    }

    {
        TaskVariantRegistrar registrar(NORM_INTRA_TASK_ID, "norm_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<int,norm_intra_task>(registrar, "norm_intra");
    }

    {
        TaskVariantRegistrar registrar(INNER_PRODUCT_INTER_TASK_ID, "inner_product_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<int,inner_product_inter_task>(registrar, "inner_product_inter");
    }

    {
        TaskVariantRegistrar registrar(INNER_PRODUCT_INTRA_TASK_ID, "inner_product_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<int,inner_product_intra_task>(registrar, "inner_product_intra");
    }

    {
        TaskVariantRegistrar registrar(GAXPY_INTER_TASK_ID, "gaxpy_inter");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<gaxpy_inter_task>(registrar, "gaxpy_inter");
    }

    {
        TaskVariantRegistrar registrar(GAXPY_INTRA_TASK_ID, "gaxpy_intra");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        registrar.set_leaf(true);
        Runtime::preregister_task_variant<gaxpy_intra_task>(registrar, "gaxpy_intra");
    }

    return Runtime::start(argc,argv);
}